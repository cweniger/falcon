import numpy as np
import asyncio
import sys
from dataclasses import dataclass
from enum import IntEnum
from fractions import Fraction
from pathlib import Path
from typing import List, Optional

import ray
from omegaconf import MISSING
from falcon.core.logger import Logger, set_logger, log, info, warning, error


@dataclass
class PathConfig:
    """Configuration for file-system paths."""

    graph: Optional[str] = None
    samples: Optional[str] = None
    buffer: Optional[str] = None
    imports: Optional[List[str]] = None  # directories prepended to sys.path in Ray workers


@dataclass
class BufferConfig:
    """Configuration for the rolling sample buffer."""

    min_samples: int = MISSING  # training + validation
    max_samples: int = MISSING  # training + validation
    validation_fraction: float = 0.15  # share of samples held out for validation only
    simulate_count: int = 64
    simulate_interval: float = 1.0
    simulate_chunk_size: int = 0
    simulate_when_full: bool = True
    initial_samples_path: Optional[str] = None
    snapshot_every: int = 0


class Batch:
    """Bidirectional batch - provides data and accepts feedback.

    Provides dictionary-style access to batch data and allows marking
    samples as disfavoured via the discard() method.

    Keys use flat dotted format: 'theta.value', 'theta.log_prob', 'x.value', etc.

    Example:
        for batch in dataloader:
            theta = batch['theta.value']
            logprob = batch['theta.log_prob']
            x = batch['x.value']

            # Discard low-likelihood samples
            mask = logprob < threshold
            batch.discard(mask)
    """

    def __init__(self, ids: np.ndarray, data: dict, dataset_manager):
        """Initialize batch with sample IDs, data dict, and dataset manager.

        Args:
            ids: Array of sample IDs for feedback (e.g., discard)
            data: Dictionary mapping keys to tensors {key: tensor}
            dataset_manager: Ray actor for dataset management
        """
        self._ids = ids
        self._data = data
        self._dataset_manager = dataset_manager

    def __getitem__(self, key: str):
        """Dictionary-style access: batch['theta']"""
        return self._data[key]

    def __contains__(self, key: str):
        """Check if key exists: 'theta' in batch"""
        return key in self._data

    def keys(self):
        """Return available keys."""
        return self._data.keys()

    def __len__(self):
        """Return batch size."""
        return len(self._ids)

    def discard(self, mask):
        """Mark samples as disfavoured based on boolean mask.

        Args:
            mask: Boolean array/tensor where True = discard sample
        """
        if mask is None:
            return
        if hasattr(mask, 'cpu'):
            mask = mask.cpu().numpy()
        if hasattr(mask, 'any') and mask.any():
            ids_to_discard = self._ids[mask].tolist()
            self._dataset_manager.deactivate.remote(ids_to_discard)
        elif isinstance(mask, (list, np.ndarray)) and any(mask):
            ids_to_discard = self._ids[mask].tolist()
            self._dataset_manager.deactivate.remote(ids_to_discard)

    def get(self, key: str, default=None):
        """Get value with optional default."""
        return self._data.get(key, default)

    def items(self):
        """Return key-value pairs."""
        return self._data.items()

    def values(self):
        """Return values."""
        return self._data.values()


class DatasetManager:
    """Access the DatasetManagerActor without exposing the actor interface directly."""

    def __init__(self, dataset_manager_actor):
        self.dataset_manager_actor = dataset_manager_actor

    def load_initial_samples(self):
        """Load pre-existing samples from disk. Returns number loaded."""
        return ray.get(self.dataset_manager_actor.load_initial_samples.remote())


class SampleStatus(IntEnum):
    """Lifecycle of a sample, independent of its SamplePurpose."""

    # Live samples (used by the loader of their purpose)
    ACTIVE = 0  # Regular live sample
    DISFAVOURED = 1  # Still used until evicted; can be moved to tombstone

    # Dead samples (will not be used anymore)
    TOMBSTONE = 2  # Marked for deletion when no longer referenced by any actor
    DELETED = 3  # Permanently deleted


class SamplePurpose(IntEnum):
    """Fixed at insertion and never changed, so training and validation never mix."""

    TRAINING = 0
    VALIDATION = 1


LIVE_STATUSES = [SampleStatus.ACTIVE, SampleStatus.DISFAVOURED]


@ray.remote
class DatasetManagerActor:
    def __init__(
        self,
        max_samples,
        min_samples,
        validation_fraction,
        simulate_count,
        simulate_interval,
        simulate_chunk_size,
        initial_samples_path,
        simulate_when_full,
        snapshot_every,
        log_config=None,
        snapshots_path=None,
    ):
        self.max_samples = max_samples
        self.min_samples = min_samples
        self._set_validation_fraction(validation_fraction)
        self.simulate_count = simulate_count
        self.simulate_when_full = simulate_when_full
        self.simulate_interval = simulate_interval
        self.simulate_chunk_size = simulate_chunk_size
        self.initial_samples_path = initial_samples_path
        self.snapshots_path = Path(snapshots_path) if snapshots_path else None
        self.snapshot_every = max(0, int(snapshot_every))

        # NPZ storage state
        self._sample_counter = 0

        # Store
        self.ray_store = []
        self.status = np.zeros(0, dtype=int)
        self.purpose = np.zeros(0, dtype=np.int8)
        self.ref_counts = np.zeros(0, dtype=int)

        # Create logger and set as module-level logger
        if log_config:
            logger = Logger("dataset", log_config, capture_exceptions=True)
            set_logger(logger)

        info(
            f"Dataset manager initialized | min_samples={min_samples} max_samples={max_samples} "
            f"validation_fraction={self._val_p}/{self._val_q}"
        )

        asyncio.create_task(self.monitor())

    def _set_validation_fraction(self, validation_fraction):
        """Store the validation fraction as an exact integer ratio p/q and derive the floors."""
        p, q = Fraction(validation_fraction).limit_denominator(1000).as_integer_ratio()
        if p < 1 or 2 * p > q:
            raise ValueError(
                f"buffer.validation_fraction={validation_fraction} must lie in (0, 0.5] "
                "(and be representable with denominator <= 1000)"
            )
        if self.min_samples < 2:
            raise ValueError(f"buffer.min_samples={self.min_samples} must be >= 2")
        self._val_p, self._val_q = p, q
        # Floors: number of each purpose among the first min_samples insertion ids
        self._min_validation = int(self._is_validation(np.arange(self.min_samples)).sum())
        self._min_training = self.min_samples - self._min_validation

    def _is_validation(self, ids):
        """Sample id i is validation-only iff (i * p) % q < p: exactly p in every q ids, evenly spread."""
        return (np.asarray(ids) * self._val_p) % self._val_q < self._val_p

    async def monitor(self):
        while True:
            self.garbage_collect_tombstones()
            await asyncio.sleep(10.0)

    def num_initial_samples(self):
        return self.min_samples

    def num_resims(self):
        if self.simulate_when_full:
            return self.simulate_count
        else:
            num_active_samples = int((self.status == SampleStatus.ACTIVE).sum())
            return min(
                self.simulate_count, self.max_samples - num_active_samples
            )

    def get_simulate_interval(self):
        return self.simulate_interval

    def get_simulate_chunk_size(self):
        return self.simulate_chunk_size
    
    # FIXME: Logging should happen through wandb only, and not funneled through training nodes
    def get_store_stats(self):
        # training/validation = live pool per purpose (what each loader draws from)
        live = np.isin(self.status, LIVE_STATUSES)
        stats = {
            "total_length": len(self.ray_store),
            "validation": int((live & (self.purpose == SamplePurpose.VALIDATION)).sum()),
            "training": int((live & (self.purpose == SamplePurpose.TRAINING)).sum()),
            "disfavoured": int((self.status == SampleStatus.DISFAVOURED).sum()),
            "tombstone": int((self.status == SampleStatus.TOMBSTONE).sum()),
            "deleted": int((self.status == SampleStatus.DELETED).sum()),
        }
        return stats

    def rotate_sample_buffer(self):
        """
        Rotate samples through lifecycle: ACTIVE -> DISFAVOURED -> TOMBSTONE.

        Purposes are fixed, so this only ages samples out; training and
        validation samples follow the same rules.
        """
        # 1) Per purpose, ACTIVE+DISFAVOURED should stay >= its share of min_samples
        #    Move excess disfavoured samples (oldest first) to tombstone
        for purpose, floor in (
            (SamplePurpose.TRAINING, self._min_training),
            (SamplePurpose.VALIDATION, self._min_validation),
        ):
            in_purpose = self.purpose == purpose
            ids_disfavoured = np.where(in_purpose & (self.status == SampleStatus.DISFAVOURED))[0]
            num_active = int((in_purpose & (self.status == SampleStatus.ACTIVE)).sum())
            num_to_keep = min(len(ids_disfavoured), max(0, floor - num_active))
            ids_to_tombstone = ids_disfavoured[: len(ids_disfavoured) - num_to_keep]
            self.status[ids_to_tombstone] = SampleStatus.TOMBSTONE

        # 2) Maximum number of ACTIVE samples (both purposes) is max_samples
        #    Move oldest excess to tombstone; purposes are interleaved, so the ratio is kept
        ids_active = np.where(self.status == SampleStatus.ACTIVE)[0]
        if len(ids_active) > self.max_samples:
            ids_to_tombstone = ids_active[: -self.max_samples]
            self.status[ids_to_tombstone] = SampleStatus.TOMBSTONE

    def checkout_refs(self, status, keys, max_samples=0, already_cached_ids=None, purpose=None):
        """Select samples by status (and purpose), return refs for uncached samples only.

        Increments ref_counts for new (uncached) sample IDs. The caller
        resolves these from the object store, then calls release_refs().

        Args:
            status: SampleStatus or list of SampleStatus
            keys: list of key names to retrieve
            max_samples: 0 = all samples, >0 = random subset
            already_cached_ids: np.array of IDs the caller already has cached.
                Only refs for IDs NOT in this set are returned.
            purpose: SamplePurpose to restrict to, or None for any purpose

        Returns:
            dict with:
                '_active_ids': np.array of all currently active sample IDs
                '_new_ids': np.array of IDs that need to be fetched
                key: [ObjectRef, ...] for _new_ids only
        """
        mask = np.isin(self.status, status if isinstance(status, list) else [status])
        if purpose is not None:
            mask &= self.purpose == purpose
        ids = np.where(mask)[0]
        if max_samples > 0 and len(ids) > max_samples:
            ids = np.random.choice(ids, size=max_samples, replace=False)

        if already_cached_ids is not None and len(already_cached_ids) > 0:
            new_ids = ids[~np.isin(ids, already_cached_ids)]
        else:
            new_ids = ids

        if len(new_ids) > 0:
            self.ref_counts[new_ids] += 1

        result = {'_active_ids': ids, '_new_ids': new_ids}
        for key in keys:
            refs = []
            for i in new_ids:
                try:
                    refs.append(self.ray_store[i][key])
                except KeyError as e:
                    # Provide more info in the error message
                    sample_keys = list(self.ray_store[i].keys()) if self.ray_store[i] else []
                    raise KeyError(f"Key '{key}' not found in sample {i}. Available keys: {sample_keys}. Total samples: {len(self.ray_store)}") from e
            result[key] = refs
        return result

    def release_refs(self, ids):
        """Decrement ref_counts after caller has resolved data."""
        if ids is not None and len(ids) > 0:
            self.ref_counts[ids] -= 1

    def deactivate(self, ids):
        # Get subset of ids that are currently ACTIVE, only these can be disfavoured
        ids = np.array(ids)
        if len(ids) > 0:
            ids_active = ids[self.status[ids] == SampleStatus.ACTIVE]
            self.status[ids_active] = SampleStatus.DISFAVOURED

    def _register_new_samples(self, num_new_samples):
        """Extend per-sample bookkeeping; purpose is fixed here by insertion id."""
        ids = np.arange(len(self.status), len(self.status) + num_new_samples)
        self.status = np.append(
            self.status, np.full(num_new_samples, SampleStatus.ACTIVE)
        )
        self.purpose = np.append(
            self.purpose,
            np.where(self._is_validation(ids), SamplePurpose.VALIDATION, SamplePurpose.TRAINING).astype(np.int8),
        )
        self.ref_counts = np.append(self.ref_counts, np.zeros(num_new_samples))

    def _log_buffer_stats(self):
        # n_training/n_validation = live pool per purpose (ACTIVE + DISFAVOURED, what each
        # loader draws from). n_disfavoured is the subset marked for eviction.
        stats = self.get_store_stats()
        log({
            "n_total": stats["total_length"],
            "n_validation": stats["validation"],
            "n_training": stats["training"],
            "n_disfavoured": stats["disfavoured"],
            "n_tombstone": stats["tombstone"],
            "n_deleted": stats["deleted"],
        }, prefix="buffer")
        return stats

    def append(self, data):
        """Append samples to the buffer.

        Args:
            data: List of dicts, each dict has {key: array} without batch dimension
        """
        num_new_samples = len(data)
        for sample in data:
            sample_ray_objects = {key: ray.put(value) for key, value in sample.items()}
            self.ray_store.append(sample_ray_objects)
        self._register_new_samples(num_new_samples)

        self.rotate_sample_buffer()
        self._log_buffer_stats()

        self.dump_store(data)

    def append_refs(self, sample_refs: list):
        """Append samples that are already ObjectRefs.

        Args:
            sample_refs: List of dicts, each dict has {key: ObjectRef}
        """
        num_new_samples = len(sample_refs)

        # Store refs directly - no ray.put() needed
        self.ray_store.extend(sample_refs)
        self._register_new_samples(num_new_samples)

        self.rotate_sample_buffer()
        stats = self._log_buffer_stats()

        info(
            f"Appended {num_new_samples} samples | total={stats['total_length']} "
            f"train={stats['training']} val={stats['validation']}"
        )

        # Disk dump: fetch values lazily if needed
        if self._snapshot_enabled():
            self._dump_refs(sample_refs)

    def _snapshot_enabled(self):
        return self.snapshot_every > 0 and self.snapshots_path is not None

    def _should_snapshot_next_sample(self):
        if not self._snapshot_enabled():
            return False
        self._sample_counter += 1
        return self._sample_counter % self.snapshot_every == 0

    def _dump_refs(self, sample_refs: list):
        """Fetch and dump samples to disk."""
        for ref_dict in sample_refs:
            if self._should_snapshot_next_sample():
                sample = {k: ray.get(v) for k, v in ref_dict.items()}
                self._save_sample(sample)

    def _save_sample(self, sample: dict):
        """Save a single sample as NPZ file to buffer/snapshots/."""
        self.snapshots_path.mkdir(parents=True, exist_ok=True)
        existing = sorted(self.snapshots_path.glob("*.npz"))
        next_idx = len(existing)
        sample_path = self.snapshots_path / f"{next_idx:06d}.npz"
        np.savez(sample_path, **sample)

    def dump_store(self, samples: list):
        """Store samples to disk based on snapshot_every.

        Args:
            samples: List of sample dicts to potentially store
        """
        if not self._snapshot_enabled():
            return

        for sample in samples:
            if self._should_snapshot_next_sample():
                self._save_sample(sample)

    def garbage_collect_tombstones(self):
        """
        Garbage collect tombstone samples with zero references.

        Frees Ray objects and clears store entries for samples that are
        marked as tombstones and no longer referenced by any operations.
        """
        unreferenced_tombstone_ids = np.where(
            (self.status == SampleStatus.TOMBSTONE) & (self.ref_counts <= 0)
        )[0]

        if len(unreferenced_tombstone_ids) > 0:
            for i in unreferenced_tombstone_ids:
                ray.internal.free(list(self.ray_store[i].values()))
                self.status[i] = SampleStatus.DELETED
                self.ray_store[i] = None

    def load_initial_samples(self):
        """Load pre-existing samples from NPZ sample directory. Returns number loaded.

        Expects initial_samples_path to point to a sample type directory
        (e.g. samples/prior) as produced by ``falcon sample prior``.
        NPZ keys are remapped: ``key`` -> ``key.value`` with ``key.log_prob = 0.0``.
        """
        if self.initial_samples_path is not None:
            from falcon.core.samples_reader import SampleSetReader

            reader = SampleSetReader(Path(self.initial_samples_path))
            samples = []
            for sample_dict in reader:
                remapped = {}
                for key, value in sample_dict.items():
                    if key.startswith("_"):
                        continue
                    remapped[f"{key}.value"] = value
                    remapped[f"{key}.log_prob"] = np.float64(0.0)
                samples.append(remapped)
            if samples:
                self.append(samples)
                info(f"Loaded {len(samples)} initial samples from {self.initial_samples_path}")
                if len(samples) > self.max_samples:
                    warning(
                        f"Loaded {len(samples)} initial samples but max_samples={self.max_samples}; "
                        f"{len(samples) - self.max_samples} oldest samples were tombstoned immediately. "
                        "Consider increasing buffer.max_samples."
                    )
            return len(samples)
        return 0


class CachedDataLoader:
    """Cached dataloader with pre-stacked torch tensors for fast batch sampling.

    Stores samples as contiguous torch tensors on a configurable device (CPU or
    GPU). refresh() incrementally updates: new samples fill free slots from
    evictions or are bulk-appended. Between refreshes the cache is fixed, and
    iter_batches() / sample_batch() use torch fancy indexing, which is ~5x
    faster than numpy for large arrays.

    When device='cuda', the entire buffer lives on GPU for maximum speed
    (~50x vs numpy dict cache). Falls back to CPU when GPU memory is
    insufficient.
    """

    def __init__(self, dataset_manager, keys, sample_status, max_cache_samples=0,
                 device=None, sample_purpose=None):
        import torch
        self.dataset_manager = dataset_manager
        self.keys = keys
        self.sample_status = sample_status
        self.sample_purpose = sample_purpose
        self.max_cache_samples = max_cache_samples
        self.device = torch.device(device) if device else torch.device('cpu')
        self.active_ids = np.array([], dtype=int)
        self.count = 0

        # Pre-stacked torch tensors
        self._arrays = {}       # key -> torch.Tensor, shape (capacity, ...)
        self._stacked_ids = np.array([], dtype=int)
        self._id_to_row = {}    # sample_id -> row index in stacked arrays
        self._free_rows = []    # reusable row indices from evicted samples
        self._active_rows = torch.zeros(0, dtype=torch.long, device=self.device)

    def _to_tensor(self, arr):
        """Convert numpy scalar/array to torch tensor on the configured device."""
        import torch
        return torch.as_tensor(np.array(arr)).to(self.device)

    def _checkout_and_fetch(self, active_ids_snapshot):
        """Check out the samples not cached yet and build their tensors."""
        import torch
        checkout = ray.get(
            self.dataset_manager.checkout_refs.remote(
                self.sample_status, self.keys, self.max_cache_samples,
                already_cached_ids=active_ids_snapshot,
                purpose=self.sample_purpose,
            )
        )
        new_ids = checkout['_new_ids']
        new_tensors = {}
        if len(new_ids) > 0:
            for key in self.keys:
                raw = ray.get(checkout[key])
                new_tensors[key] = torch.as_tensor(np.stack(raw)).to(self.device)
            ray.get(self.dataset_manager.release_refs.remote(new_ids))
        return new_tensors, checkout

    def _apply_fetch(self, new_tensors, checkout):
        """Apply pre-built tensors to the cache with indexed scatter and torch.cat."""
        import torch
        active_ids = checkout['_active_ids']
        new_ids = checkout['_new_ids']

        # Evict stale samples: mark their rows as free
        active_set = set(active_ids.tolist())
        for sid in list(self._id_to_row.keys()):
            if sid not in active_set:
                self._free_rows.append(self._id_to_row.pop(sid))

        # Insert new samples
        if len(new_ids) == 0:
            pass
        elif len(self._arrays) == 0:
            # First sync: use pre-built tensors directly
            for key in self.keys:
                self._arrays[key] = new_tensors[key]
            self._stacked_ids = np.array(new_ids)
            for i, sid in enumerate(new_ids):
                self._id_to_row[sid] = i
        else:
            # Match dtypes: background thread may produce different precision
            for key in self.keys:
                if new_tensors[key].dtype != self._arrays[key].dtype:
                    new_tensors[key] = new_tensors[key].to(self._arrays[key].dtype)

            # Scatter into free slots (batched indexed assignment)
            n_free = min(len(new_ids), len(self._free_rows))
            if n_free > 0:
                free_rows = [self._free_rows.pop() for _ in range(n_free)]
                row_idx = torch.tensor(free_rows, dtype=torch.long, device=self.device)
                for key in self.keys:
                    self._arrays[key][row_idx] = new_tensors[key][:n_free]
                for i, row in enumerate(free_rows):
                    sid = new_ids[i]
                    self._stacked_ids[row] = sid
                    self._id_to_row[sid] = row

            # Bulk-append remainder
            if n_free < len(new_ids):
                for key in self.keys:
                    self._arrays[key] = torch.cat(
                        [self._arrays[key], new_tensors[key][n_free:]], dim=0
                    )
                base_row = len(self._stacked_ids)
                self._stacked_ids = np.concatenate(
                    [self._stacked_ids, np.array(new_ids[n_free:])]
                )
                for j, sid in enumerate(new_ids[n_free:]):
                    self._id_to_row[sid] = base_row + j

        self.active_ids = active_ids

        # Build index of active rows for sampling
        self._active_rows = torch.tensor(
            list(self._id_to_row.values()),
            dtype=torch.long, device=self.device,
        )
        self.count = len(self._active_rows)

    def refresh(self):
        """Bring the cache up to date with the buffer; it then stays fixed until the next refresh."""
        new_tensors, checkout = self._checkout_and_fetch(self.active_ids.copy())
        self._apply_fetch(new_tensors, checkout)

    def _batch(self, rows):
        """Batch object for the given row indices of the stacked arrays."""
        ids = self._stacked_ids[rows.cpu().numpy()]
        data = {key: arr[rows] for key, arr in self._arrays.items()}
        return Batch(ids, data, self.dataset_manager)

    def sample_batch(self, batch_size):
        """Random mini-batch as a Batch object."""
        import torch
        idx = torch.randint(0, self.count, (batch_size,), device=self.device)
        return self._batch(self._active_rows[idx])

    def iter_batches(self, batch_size, shuffle=False, drop_last=False):
        """Yield Batch objects that together cover every cached sample once.

        Args:
            batch_size: Samples per batch.
            shuffle: Visit samples in random order instead of cache order.
            drop_last: Skip a final partial batch, unless it is the only batch.
        """
        import torch
        rows = self._active_rows
        if shuffle:
            rows = rows[torch.randperm(self.count, device=self.device)]
        num_full = self.count // batch_size
        num_batches = num_full if drop_last and num_full > 0 else -(-self.count // batch_size)
        for b in range(num_batches):
            yield self._batch(rows[b * batch_size:(b + 1) * batch_size])


class BufferView:
    """View into the sample buffer for estimator training.

    Passed to RoundTrainer.run(), which requests cached dataloaders with specific keys.
    Keys use flat dotted format: 'theta.value', 'theta.log_prob', 'x.value', etc.

    Example:
        def run(self, buffer: BufferView):
            keys = self.model.batch_keys()
            train_cache = buffer.cached_loader(keys)
            val_cache = buffer.cached_val_loader(keys)
            train_cache.refresh()
            val_cache.refresh()
            for batch in train_cache.iter_batches(batch_size, shuffle=True):
                theta = batch[f'{self.theta_key}.value']
                ...
    """

    def __init__(self, dataset_manager, cache_device=None):
        """Initialize buffer view.

        Args:
            dataset_manager: Ray actor for dataset management
            cache_device: Device for cached tensors ('cpu', 'cuda', or None for cpu).
        """
        self._dataset_manager = dataset_manager
        self._cache_device = cache_device

    def cached_loader(self, keys, max_cache_samples=0):
        """Create a training dataloader with cached tensors on the configured device."""
        return CachedDataLoader(
            self._dataset_manager, keys,
            sample_status=LIVE_STATUSES,
            sample_purpose=SamplePurpose.TRAINING,
            max_cache_samples=max_cache_samples,
            device=self._cache_device,
        )

    def cached_val_loader(self, keys, max_cache_samples=0):
        """Create a validation dataloader with cached tensors on the configured device."""
        return CachedDataLoader(
            self._dataset_manager, keys,
            sample_status=LIVE_STATUSES,
            sample_purpose=SamplePurpose.VALIDATION,
            max_cache_samples=max_cache_samples,
            device=self._cache_device,
        )

    def get_stats(self):
        """Get buffer statistics (total samples, etc.)."""
        return ray.get(self._dataset_manager.get_store_stats.remote())


def get_ray_dataset_manager(
    config: BufferConfig,
    snapshots_path=None,
    log_config=None,
):
    from omegaconf import OmegaConf
    cfg = OmegaConf.to_container(config, resolve=True) if not isinstance(config, dict) else config
    dataset_manager_actor = DatasetManagerActor.remote(
        **cfg,
        snapshots_path=snapshots_path,
        log_config=log_config,
    )
    return DatasetManager(dataset_manager_actor)
