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

    Example:
        for batch in dataloader:
            theta = batch['z']
            logprob = batch['z.logprob']
            x = batch['x']

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
            result[key] = [self.ray_store[i][key] for i in new_ids]
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
    """CPU-cached dataloader for fast training access.

    Created once, holds samples as numpy arrays in a local dict.
    sync() pulls current state from DatasetManager via decentralized
    object store access. sample_batch() returns a Batch object.
    """

    def __init__(self, dataset_manager, keys, sample_status, max_cache_samples=0):
        self.dataset_manager = dataset_manager
        self.keys = keys
        self.sample_status = sample_status
        self.max_cache_samples = max_cache_samples
        self.cache = {}  # sample_id -> {key: np.ndarray}
        self.active_ids = np.array([], dtype=int)
        self._id_list = []
        self.count = 0

    def sync(self):
        """Incremental sync: only fetch new samples, evict stale ones."""
        checkout = ray.get(
            self.dataset_manager.checkout_refs.remote(
                self.sample_status, self.keys, self.max_cache_samples,
                already_cached_ids=self.active_ids,
            )
        )
        active_ids = checkout['_active_ids']
        new_ids = checkout['_new_ids']

        if len(new_ids) > 0:
            new_data = {}
            for key in self.keys:
                refs = checkout[key]
                arrays = ray.get(refs)
                new_data[key] = arrays

            for i, sid in enumerate(new_ids):
                self.cache[sid] = {key: new_data[key][i] for key in self.keys}

            ray.get(self.dataset_manager.release_refs.remote(new_ids))

        # Evict samples no longer active
        active_set = set(active_ids.tolist())
        stale = [sid for sid in self.cache if sid not in active_set]
        for sid in stale:
            del self.cache[sid]

        self.active_ids = active_ids
        self._id_list = list(active_set & set(self.cache.keys()))
        self.count = len(self._id_list)

    def sample_batch(self, batch_size):
        """Random mini-batch as a Batch object."""
        idx = np.random.randint(0, self.count, size=batch_size)
        selected = [self._id_list[i] for i in idx]

        ids = np.array(selected)
        data = {
            key: np.stack([self.cache[sid][key] for sid in selected])
            for key in self.keys
        }
        return Batch(ids, data, self.dataset_manager)


class BufferView:
    """View into the sample buffer for estimator training.

    Passed to estimator.train() - estimator requests cached dataloaders with specific keys.

    Example:
        async def train(self, buffer: BufferView):
            keys = [self.theta_key, f"{self.theta_key}.logprob", *self.condition_keys]
            train_cache = buffer.cached_loader(keys)
            val_cache = buffer.cached_val_loader(keys)
            train_cache.sync()
            val_cache.sync()
            for step in range(steps_per_epoch):
                batch = train_cache.sample_batch(batch_size)
                theta = batch[self.theta_key]
                ...
    """

    def __init__(self, dataset_manager):
        """Initialize buffer view.

        Args:
            dataset_manager: Ray actor for dataset management
        """
        self._dataset_manager = dataset_manager

    def cached_loader(self, keys, max_cache_samples=0):
        """Create a CPU-cached training dataloader (created once, syncs periodically)."""
        return CachedDataLoader(
            self._dataset_manager, keys,
            sample_status=[SampleStatus.TRAINING, SampleStatus.DISFAVOURED],
            max_cache_samples=max_cache_samples,
        )

    def cached_val_loader(self, keys, max_cache_samples=0):
        """Create a CPU-cached validation dataloader."""
        return CachedDataLoader(
            self._dataset_manager, keys,
            sample_status=SampleStatus.VALIDATION,
            max_cache_samples=max_cache_samples,
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
        keep_resampling=keep_resampling,
        initial_samples_path=initial_samples_path,
        buffer_path=buffer_path,
        store_fraction=store_fraction,
        log_config=log_config,
    )
    dataset_manager = DatasetManager(dataset_manager_actor)
    return dataset_manager
