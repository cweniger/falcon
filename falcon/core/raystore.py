import numpy as np
import asyncio
import sys
from enum import IntEnum
from datetime import datetime
from pathlib import Path

import ray
from falcon.core.logger import Logger, set_logger, log, error


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
    # Live samples (used for validation/training)
    VALIDATION = 0  # New samples for validation
    TRAINING = 1  # Older samples for training
    DISFAVOURED = 2  # Can be moved to tombstone

    # Dead samples (will not be used anymore)
    TOMBSTONE = 3  # Marked for deletion when no longer referenced by any actor
    DELETED = 4  # Permanently deleted


@ray.remote(name="DatasetManager")
class DatasetManagerActor:
    def __init__(
        self,
        max_training_samples=None,  # TODO: Maximum number of simulations to store
        min_training_samples=None,  # TODO: Minimum number of simulations to train on
        validation_window_size=None,  # TODO: Number of sliding validation sims
        resample_batch_size=256,
        resample_interval=5,
        simulate_chunk_size=0,
        initial_samples_path=None,
        keep_resampling=True,
        buffer_path=None,
        store_fraction=0.0,
        log_config=None,
    ):
        self.max_training_samples = max_training_samples
        self.min_training_samples = min_training_samples
        self.validation_window_size = validation_window_size
        self.resample_batch_size = resample_batch_size
        self.keep_resampling = keep_resampling
        self.resample_interval = resample_interval
        self.simulate_chunk_size = simulate_chunk_size
        self.initial_samples_path = initial_samples_path
        self.buffer_path = Path(buffer_path) if buffer_path else None
        self.store_fraction = store_fraction

        # NPZ storage state
        self._sample_counter = 0
        self._session_dir = None

        # Store
        self.ray_store = []
        self.status = np.zeros(0, dtype=int)
        self.ref_counts = np.zeros(0, dtype=int)

        # Create logger and set as module-level logger
        if log_config:
            logger = Logger("dataset", log_config, capture_exceptions=True)
            set_logger(logger)

        asyncio.create_task(self.monitor())

    async def monitor(self):
        while True:
            self.garbage_collect_tombstones()
            await asyncio.sleep(10.0)

    def num_initial_samples(self):
        return self.min_training_samples + self.validation_window_size

    def num_resims(self):
        if self.keep_resampling:
            return self.resample_batch_size
        else:
            num_train_samples = sum(self.status == SampleStatus.TRAINING)
            return min(
                self.resample_batch_size, self.max_training_samples - num_train_samples
            )

    def get_resample_interval(self):
        return self.resample_interval

    def get_simulate_chunk_size(self):
        return self.simulate_chunk_size
    
    # FIXME: Logging should happen through wandb only, and not funneled through training nodes
    def get_store_stats(self):
        stats = {
            "total_length": len(self.ray_store),
            "validation": sum(self.status == SampleStatus.VALIDATION),
            "training": sum(self.status == SampleStatus.TRAINING),
            "disfavoured": sum(self.status == SampleStatus.DISFAVOURED),
            "tombstone": sum(self.status == SampleStatus.TOMBSTONE),
            "deleted": sum(self.status == SampleStatus.DELETED),
        }
        return stats

    def rotate_sample_buffer(self):
        """
        Rotate samples through lifecycle: VAL -> TRAIN -> DISFAVOURED -> TOMBSTONE.

        Keeps most recent samples for validation, older samples for training,
        and marks oldest samples as disfavoured and then tombstone for deletion.
        """
        # 1) Maximum number of VALIDATION samples should be validation_window_size
        #    Move excess validation samples to training
        ids_validation = np.where(self.status == SampleStatus.VALIDATION)[0]
        num_val_samples = len(ids_validation)
        if num_val_samples > self.validation_window_size:
            ids_to_train = ids_validation[: -self.validation_window_size]
            self.status[ids_to_train] = SampleStatus.TRAINING

        # 2) Minimum number of TRAINING+DISFAVOURED samples should be min_training_samples
        #    Move excess disfavoured samples to tombstone
        ids_disfavoured = np.where(self.status == SampleStatus.DISFAVOURED)[0]
        num_train_samples = sum(self.status == SampleStatus.TRAINING)
        num_disfavoured_samples = len(ids_disfavoured)
        num_disfavoured_samples_to_keep = max(
            0, self.min_training_samples - num_train_samples
        )
        if num_disfavoured_samples_to_keep == 0:
            self.status[ids_disfavoured] = SampleStatus.TOMBSTONE
        elif num_disfavoured_samples > num_disfavoured_samples_to_keep:
            ids_to_tombstone = ids_disfavoured[:-num_disfavoured_samples_to_keep]
            self.status[ids_to_tombstone] = SampleStatus.TOMBSTONE

        # 3) Maximum number of TRAINING samples is max_training_samples
        #    Move excess training samples to tombstone
        ids_training = np.where(self.status == SampleStatus.TRAINING)[0]
        num_train_samples = len(ids_training)
        if num_train_samples > self.max_training_samples:
            ids_to_tombstone = ids_training[: -self.max_training_samples]
            self.status[ids_to_tombstone] = SampleStatus.TOMBSTONE

    def checkout_refs(self, status, keys, max_samples=0, already_cached_ids=None):
        """Select samples by status, return refs for uncached samples only.

        Increments ref_counts for new (uncached) sample IDs. The caller
        resolves these from the object store, then calls release_refs().

        Args:
            status: SampleStatus or list of SampleStatus
            keys: list of key names to retrieve
            max_samples: 0 = all samples, >0 = random subset
            already_cached_ids: np.array of IDs the caller already has cached.
                Only refs for IDs NOT in this set are returned.

        Returns:
            dict with:
                '_active_ids': np.array of all currently active sample IDs
                '_new_ids': np.array of IDs that need to be fetched
                key: [ObjectRef, ...] for _new_ids only
        """
        if isinstance(status, list):
            ids = np.where(np.isin(self.status, status))[0]
        else:
            ids = np.where(self.status == status)[0]
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
        # Get subset of ids that are currently TRAINING, only these can be disfavoured
        ids = np.array(ids)
        if len(ids) > 0:
            ids_training = ids[self.status[ids] == SampleStatus.TRAINING]
            self.status[ids_training] = SampleStatus.DISFAVOURED

    def append(self, data):
        """Append samples to the buffer.

        Args:
            data: List of dicts, each dict has {key: array} without batch dimension
        """
        num_new_samples = len(data)
        for sample in data:
            sample_ray_objects = {key: ray.put(value) for key, value in sample.items()}
            self.ray_store.append(sample_ray_objects)
        self.status = np.append(
            self.status, np.full(num_new_samples, SampleStatus.VALIDATION)
        )
        self.ref_counts = np.append(self.ref_counts, np.zeros(num_new_samples))

        self.rotate_sample_buffer()

        # Log buffer statistics
        log({
            "n_total": len(self.ray_store),
            "n_validation": int(sum(self.status == SampleStatus.VALIDATION)),
            "n_training": int(sum(self.status == SampleStatus.TRAINING)),
            "n_disfavoured": int(sum(self.status == SampleStatus.DISFAVOURED)),
            "n_tombstone": int(sum(self.status == SampleStatus.TOMBSTONE)),
            "n_deleted": int(sum(self.status == SampleStatus.DELETED)),
        }, prefix="buffer")

        self.dump_store(data)

    def append_refs(self, sample_refs: list):
        """Append samples that are already ObjectRefs.

        Args:
            sample_refs: List of dicts, each dict has {key: ObjectRef}
        """
        num_new_samples = len(sample_refs)

        # Store refs directly - no ray.put() needed
        self.ray_store.extend(sample_refs)

        self.status = np.append(
            self.status, np.full(num_new_samples, SampleStatus.VALIDATION)
        )
        self.ref_counts = np.append(self.ref_counts, np.zeros(num_new_samples))

        self.rotate_sample_buffer()

        # Log buffer statistics
        log({
            "n_total": len(self.ray_store),
            "n_validation": int(sum(self.status == SampleStatus.VALIDATION)),
            "n_training": int(sum(self.status == SampleStatus.TRAINING)),
            "n_disfavoured": int(sum(self.status == SampleStatus.DISFAVOURED)),
            "n_tombstone": int(sum(self.status == SampleStatus.TOMBSTONE)),
            "n_deleted": int(sum(self.status == SampleStatus.DELETED)),
        }, prefix="buffer")

        # Disk dump: fetch values lazily if needed
        if self.store_fraction > 0 and self.buffer_path is not None:
            self._dump_refs(sample_refs)

    def _dump_refs(self, sample_refs: list):
        """Fetch and dump samples to disk."""
        interval = max(1, int(1.0 / self.store_fraction))

        for ref_dict in sample_refs:
            self._sample_counter += 1
            if self._sample_counter % interval == 0:
                sample = {k: ray.get(v) for k, v in ref_dict.items()}
                self._save_sample(sample)

    def _save_sample(self, sample: dict):
        """Save a single sample as NPZ file."""
        if self._session_dir is None:
            # Create session directory on first write
            timestamp = datetime.now().strftime("%y%m%d-%H%M")
            self._session_dir = self.buffer_path / timestamp
            self._session_dir.mkdir(parents=True, exist_ok=True)

        # Use counter for filename (0-indexed from session start)
        sample_idx = self._sample_counter - 1  # counter already incremented
        sample_path = self._session_dir / f"{sample_idx:06d}.npz"
        np.savez(sample_path, **sample)

    def dump_store(self, samples: list):
        """Store samples to disk based on store_fraction.

        Args:
            samples: List of sample dicts to potentially store
        """
        if self.store_fraction <= 0 or self.buffer_path is None:
            return

        interval = max(1, int(1.0 / self.store_fraction))

        for sample in samples:
            self._sample_counter += 1
            if self._sample_counter % interval == 0:
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
        """Load pre-existing samples from disk. Returns number loaded."""
        if self.initial_samples_path is not None:
            initial_samples = joblib.load(self.initial_samples_path)
            if len(initial_samples) > 0:
                self.append(initial_samples)
            return len(initial_samples)
        return 0

    # TODO: Currently not used anywhere, add tests?
    def shutdown(self):
        pass


class CachedDataLoader:
    """Cached dataloader with pre-stacked torch tensors for fast batch sampling.

    Stores samples as contiguous torch tensors on a configurable device (CPU or
    GPU). sync() incrementally updates: new samples fill free slots from
    evictions or are bulk-appended. sample_batch() uses torch fancy indexing,
    which is ~5x faster than numpy for large arrays.

    Data fetching (ray.get) runs in a background thread so the training event
    loop is not blocked. New data becomes available one sync() call later.

    When device='cuda', the entire buffer lives on GPU for maximum speed
    (~50x vs numpy dict cache). Falls back to CPU when GPU memory is
    insufficient.
    """

    def __init__(self, dataset_manager, keys, sample_status, max_cache_samples=0,
                 device=None):
        import torch
        from concurrent.futures import ThreadPoolExecutor
        self.dataset_manager = dataset_manager
        self.keys = keys
        self.sample_status = sample_status
        self.max_cache_samples = max_cache_samples
        self.device = torch.device(device) if device else torch.device('cpu')
        self.active_ids = np.array([], dtype=int)
        self.count = 0

        # Pre-stacked torch tensors
        self._arrays = {}       # key -> torch.Tensor, shape (capacity, ...)
        self._stacked_ids = np.array([], dtype=int)
        self._id_to_row = {}    # sample_id -> row index in stacked arrays
        self._free_rows = []    # reusable row indices from evicted samples

        # Background fetch thread
        self._fetch_executor = ThreadPoolExecutor(max_workers=1)
        self._pending_fetch = None  # Future -> (new_data, checkout)

    def _to_tensor(self, arr):
        """Convert numpy scalar/array to torch tensor on the configured device."""
        import torch
        return torch.as_tensor(np.array(arr)).to(self.device)

    def _fetch_data(self, checkout):
        """Fetch new sample data from Ray object store (runs in background thread)."""
        new_data = {}
        for key in self.keys:
            new_data[key] = ray.get(checkout[key])
        ray.get(self.dataset_manager.release_refs.remote(checkout['_new_ids']))
        return new_data, checkout

    def _apply_fetch(self, new_data, checkout):
        """Apply fetched data to the cache (runs on main thread at epoch boundary)."""
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
            # First sync: bulk-build stacked tensors
            for key in self.keys:
                self._arrays[key] = self._to_tensor(np.stack(new_data[key]))
            self._stacked_ids = np.array(new_ids)
            for i, sid in enumerate(new_ids):
                self._id_to_row[sid] = i
        else:
            # Incremental: fill free slots first, then bulk-append remainder
            idx = 0
            while idx < len(new_ids) and self._free_rows:
                row = self._free_rows.pop()
                sid = new_ids[idx]
                for key in self.keys:
                    self._arrays[key][row] = self._to_tensor(new_data[key][idx])
                self._stacked_ids[row] = sid
                self._id_to_row[sid] = row
                idx += 1

            if idx < len(new_ids):
                # Bulk-append remaining
                for key in self.keys:
                    tail = self._to_tensor(np.stack(new_data[key][idx:]))
                    self._arrays[key] = torch.cat(
                        [self._arrays[key], tail], dim=0
                    )
                base_row = len(self._stacked_ids)
                self._stacked_ids = np.concatenate(
                    [self._stacked_ids, np.array(new_ids[idx:])]
                )
                for j, sid in enumerate(new_ids[idx:]):
                    self._id_to_row[sid] = base_row + j

        self.active_ids = active_ids

        # Build index of active rows for sampling
        self._active_rows = torch.tensor(
            [self._id_to_row[sid] for sid in self._id_to_row],
            dtype=torch.long, device=self.device,
        )
        self.count = len(self._active_rows)

    def sync(self):
        """Incremental sync with background data fetching.

        Heavy ray.get calls run in a background thread. New data is applied
        on the next sync() call, giving a 1-epoch delay but never blocking
        the training event loop for bulk data transfer.
        """
        # 1. Apply any completed background fetch
        if self._pending_fetch is not None:
            new_data, checkout = self._pending_fetch.result()
            self._apply_fetch(new_data, checkout)
            self._pending_fetch = None

        # 2. Check for new data and start background fetch
        checkout = ray.get(
            self.dataset_manager.checkout_refs.remote(
                self.sample_status, self.keys, self.max_cache_samples,
                already_cached_ids=self.active_ids,
            )
        )
        new_ids = checkout['_new_ids']

        if len(new_ids) > 0:
            if len(self._arrays) == 0:
                # First sync: must block to have data for training
                new_data, checkout = self._fetch_data(checkout)
                self._apply_fetch(new_data, checkout)
            else:
                # Subsequent syncs: fetch in background
                self._pending_fetch = self._fetch_executor.submit(
                    self._fetch_data, checkout
                )
        else:
            # No new data, just update evictions
            active_ids = checkout['_active_ids']
            active_set = set(active_ids.tolist())
            for sid in list(self._id_to_row.keys()):
                if sid not in active_set:
                    self._free_rows.append(self._id_to_row.pop(sid))
            self.active_ids = active_ids
            import torch
            self._active_rows = torch.tensor(
                [self._id_to_row[sid] for sid in self._id_to_row],
                dtype=torch.long, device=self.device,
            )
            self.count = len(self._active_rows)

    def sample_batch(self, batch_size):
        """Random mini-batch as a Batch object."""
        import torch
        idx = torch.randint(0, self.count, (batch_size,), device=self.device)
        rows = self._active_rows[idx]
        ids = self._stacked_ids[rows.cpu().numpy()] if self.device.type != 'cpu' else self._stacked_ids[rows.numpy()]
        data = {key: arr[rows] for key, arr in self._arrays.items()}
        return Batch(ids, data, self.dataset_manager)


class BufferView:
    """View into the sample buffer for estimator training.

    Passed to estimator.train() - estimator requests cached dataloaders with specific keys.
    Keys use flat dotted format: 'theta.value', 'theta.log_prob', 'x.value', etc.

    Example:
        async def train(self, buffer: BufferView):
            keys = [f"{self.theta_key}.value", f"{self.theta_key}.log_prob",
                    *[f"{k}.value" for k in self.condition_keys]]
            train_cache = buffer.cached_loader(keys)
            val_cache = buffer.cached_val_loader(keys)
            train_cache.sync()
            val_cache.sync()
            for step in range(steps_per_epoch):
                batch = train_cache.sample_batch(batch_size)
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
            sample_status=[SampleStatus.TRAINING, SampleStatus.DISFAVOURED],
            max_cache_samples=max_cache_samples,
            device=self._cache_device,
        )

    def cached_val_loader(self, keys, max_cache_samples=0):
        """Create a validation dataloader with cached tensors on the configured device."""
        return CachedDataLoader(
            self._dataset_manager, keys,
            sample_status=SampleStatus.VALIDATION,
            max_cache_samples=max_cache_samples,
            device=self._cache_device,
        )

    def get_stats(self):
        """Get buffer statistics (total samples, etc.)."""
        return ray.get(self._dataset_manager.get_store_stats.remote())


def get_ray_dataset_manager(
    min_training_samples=None,
    max_training_samples=None,
    validation_window_size=None,
    resample_batch_size=64,
    resample_interval=5,
    simulate_chunk_size=0,
    initial_samples_path=None,
    keep_resampling=True,
    buffer_path=None,
    store_fraction=0.0,
    log_config=None,
):
    dataset_manager_actor = DatasetManagerActor.remote(
        min_training_samples=min_training_samples,
        max_training_samples=max_training_samples,
        validation_window_size=validation_window_size,
        resample_batch_size=resample_batch_size,
        resample_interval=resample_interval,
        simulate_chunk_size=simulate_chunk_size,
        keep_resampling=keep_resampling,
        initial_samples_path=initial_samples_path,
        buffer_path=buffer_path,
        store_fraction=store_fraction,
        log_config=log_config,
    )
    dataset_manager = DatasetManager(dataset_manager_actor)
    return dataset_manager
