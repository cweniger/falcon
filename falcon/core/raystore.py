import numpy as np
import asyncio
from enum import IntEnum
import joblib

import ray
from torch.utils.data import IterableDataset

from falcon.core.logging import initialize_logging_for, log


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

    def initialize_samples(self, deployed_graph):
        ray.get(self.dataset_manager_actor.initialize_samples.remote(deployed_graph))


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
        dump_config=None,
    ):
        self.max_training_samples = max_training_samples
        self.min_training_samples = min_training_samples
        self.validation_window_size = validation_window_size
        self.resample_batch_size = resample_batch_size
        self.keep_resampling = keep_resampling
        self.resample_interval = resample_interval
        self.initial_samples_path = initial_samples_path
        self.dump_config = dump_config

        # Store
        self.ray_store = []
        self.status = np.zeros(0, dtype=int)
        self.ref_counts = np.zeros(0, dtype=int)

        initialize_logging_for("Dataset")

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

    def get_samples_by_status(self, status):
        if isinstance(status, list):
            status_ids = np.where(np.isin(self.status, status))[0]
        else:
            status_ids = np.where(self.status == status)[0]
        status_samples = [self.ray_store[i] for i in status_ids]
        self.ref_counts[status_ids] += 1
        return status_samples, status_ids

    def release_samples(self, ids):
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

        log({"buffer:n_total": len(self.ray_store)})
        log({"buffer:n_validation": sum(self.status == SampleStatus.VALIDATION)})
        log({"buffer:n_training": sum(self.status == SampleStatus.TRAINING)})
        log({"buffer:n_disfavoured": sum(self.status == SampleStatus.DISFAVOURED)})
        log({"buffer:n_tombstone": sum(self.status == SampleStatus.TOMBSTONE)})
        log({"buffer:n_deleted": sum(self.status == SampleStatus.DELETED)})

        self.dump_store()

    def dump_store(self):
        # FIXME: Samples should be stored within single file, and reasonable directory
        if self.dump_config and self.dump_config.enabled:
            dump_path = self.dump_config.path.format(step=len(self.ray_store))
            sample_data = self.ray_store[-1]
            sample_data = {key: ray.get(value) for key, value in sample_data.items()}
            joblib.dump(sample_data, dump_path)

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

    def get_train_dataset_view(self, keys, filter=None):
        dataset_train = DatasetView(
            keys,
            filter=filter,
            sample_status=[SampleStatus.TRAINING, SampleStatus.DISFAVOURED],
        )
        return dataset_train

    def get_val_dataset_view(self, keys, filter=None):
        dataset_val = DatasetView(
            keys, filter=filter, sample_status=SampleStatus.VALIDATION
        )
        return dataset_val

    def get_train_batch_dataset_view(self, keys, filter=None):
        """Get training dataset view that yields dictionaries for Batch collation."""
        dataset_train = BatchDatasetView(
            keys,
            filter=filter,
            sample_status=[SampleStatus.TRAINING, SampleStatus.DISFAVOURED],
        )
        return dataset_train

    def get_val_batch_dataset_view(self, keys, filter=None):
        """Get validation dataset view that yields dictionaries for Batch collation."""
        dataset_val = BatchDatasetView(
            keys, filter=filter, sample_status=SampleStatus.VALIDATION
        )
        return dataset_val

    def initialize_samples(self, deployed_graph):
        num_initial_samples = self.num_initial_samples()
        if self.initial_samples_path is not None:
            # Load initial samples from the specified path (already list of dicts)
            initial_samples = joblib.load(self.initial_samples_path)
            num_loaded_samples = len(initial_samples)
            if num_loaded_samples > 0:
                self.append(initial_samples)
        else:
            num_loaded_samples = 0
        if num_initial_samples > num_loaded_samples:
            # deployed_graph.sample() returns dict-of-arrays, convert to list-of-dicts
            samples_batched = deployed_graph.sample(num_initial_samples - num_loaded_samples)
            n = samples_batched[list(samples_batched.keys())[0]].shape[0]
            samples = [{k: v[i] for k, v in samples_batched.items()} for i in range(n)]
            self.append(samples)

    # TODO: Currently not used anywhere, add tests?
    def shutdown(self):
        pass


class DatasetView(IterableDataset):
    """Dataset view that yields samples as tuples (legacy format).

    Yields:
        Tuple of (index, value1, value2, ...) where values correspond to keylist order.
    """

    def __init__(self, keylist, filter=None, sample_status=SampleStatus.TRAINING):
        self.dataset_manager = ray.get_actor("DatasetManager")
        self.keylist = keylist
        self.filter = filter
        self.sample_status = sample_status

    def __iter__(self):
        active_samples, active_ids = ray.get(
            self.dataset_manager.get_samples_by_status.remote(self.sample_status)
        )

        perm = np.random.permutation(len(active_samples))

        if self.sample_status == SampleStatus.TRAINING:
            log({"dataset:train_size": len(perm)})
        elif self.sample_status == SampleStatus.VALIDATION:
            log({"dataset:val_size": len(perm)})
        else:
            log({"dataset:active_size": len(perm)})

        for i in perm:
            try:
                sample = [ray.get(active_samples[i][key]) for key in self.keylist]
            except Exception as e:
                print(f"Error retrieving sample {i}: {e}")
                continue
            if self.filter is not None:  # Online evaluation
                sample = self.filter(sample)
            index = active_ids[i]
            yield (index, *sample)

        ray.get(self.dataset_manager.release_samples.remote(active_ids))


class BatchDatasetView(IterableDataset):
    """Dataset view that yields samples as dictionaries for Batch collation.

    Works with batch_collate_fn to produce Batch objects with dictionary access.

    Yields:
        Tuple of (index, {key: value, ...}) where keys are from keylist.
    """

    def __init__(self, keylist, filter=None, sample_status=SampleStatus.TRAINING):
        self.dataset_manager = ray.get_actor("DatasetManager")
        self.keylist = keylist
        self.filter = filter
        self.sample_status = sample_status

    def __iter__(self):
        active_samples, active_ids = ray.get(
            self.dataset_manager.get_samples_by_status.remote(self.sample_status)
        )

        perm = np.random.permutation(len(active_samples))

        if self.sample_status == SampleStatus.TRAINING:
            log({"dataset:train_size": len(perm)})
        elif self.sample_status == SampleStatus.VALIDATION:
            log({"dataset:val_size": len(perm)})
        else:
            log({"dataset:active_size": len(perm)})

        for i in perm:
            try:
                sample_dict = {
                    key: ray.get(active_samples[i][key]) for key in self.keylist
                }
            except Exception as e:
                print(f"Error retrieving sample {i}: {e}")
                continue
            if self.filter is not None:
                sample_dict = self.filter(sample_dict)
            index = active_ids[i]
            yield (index, sample_dict)

        ray.get(self.dataset_manager.release_samples.remote(active_ids))


def batch_collate_fn(dataset_manager):
    """Create a collate function that produces Batch objects.

    Args:
        dataset_manager: Ray actor for dataset management (for discard functionality)

    Returns:
        Collate function for use with PyTorch DataLoader

    Example:
        dataset = BatchDatasetView(keys, sample_status=SampleStatus.TRAINING)
        dataloader = DataLoader(
            dataset,
            batch_size=128,
            collate_fn=batch_collate_fn(dataset_manager)
        )
        for batch in dataloader:
            theta = batch['z']
            batch.discard(mask)
    """

    def collate(samples):
        """Collate samples into a Batch object with numpy arrays.

        Returns numpy arrays for framework-agnostic data transport.
        Estimators convert to their framework (PyTorch, JAX, etc.) as needed.
        """
        # samples is list of (index, {key: value}) tuples
        ids = np.array([s[0] for s in samples])

        # Stack values for each key as numpy arrays
        data = {}
        keys = samples[0][1].keys()
        for key in keys:
            values = [s[1][key] for s in samples]
            if isinstance(values[0], np.ndarray):
                data[key] = np.stack(values)
            elif hasattr(values[0], 'numpy'):
                # torch tensor - convert to numpy
                data[key] = np.stack([v.numpy() for v in values])
            else:
                # Scalar or other - convert to numpy array
                data[key] = np.array(values)

        return Batch(ids, data, dataset_manager)

    return collate


class BufferView:
    """View into the sample buffer for estimator training.

    Passed to estimator.train() - estimator requests dataloaders with specific keys.

    Example:
        async def train(self, buffer: BufferView):
            keys = [self.theta_key, f"{self.theta_key}.logprob", *self.condition_keys]
            train_loader = buffer.train_loader(keys, batch_size=self.batch_size)
            val_loader = buffer.val_loader(keys, batch_size=self.batch_size)
            for batch in train_loader:
                theta = batch[self.theta_key]
                ...
    """

    def __init__(self, dataset_manager):
        """Initialize buffer view.

        Args:
            dataset_manager: Ray actor for dataset management
        """
        self._dataset_manager = dataset_manager

    def train_loader(self, keys: list, batch_size: int = 128, **kwargs):
        """Create training dataloader with specified keys.

        Args:
            keys: List of keys to include in batches (e.g., ['z', 'z.logprob', 'x'])
            batch_size: Batch size for dataloader
            **kwargs: Additional arguments passed to DataLoader

        Returns:
            DataLoader yielding Batch objects with numpy arrays
        """
        from torch.utils.data import DataLoader

        dataset = ray.get(
            self._dataset_manager.get_train_batch_dataset_view.remote(keys, filter=None)
        )
        collate_fn = batch_collate_fn(self._dataset_manager)
        return DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, **kwargs)

    def val_loader(self, keys: list, batch_size: int = 128, **kwargs):
        """Create validation dataloader with specified keys.

        Args:
            keys: List of keys to include in batches (e.g., ['z', 'z.logprob', 'x'])
            batch_size: Batch size for dataloader
            **kwargs: Additional arguments passed to DataLoader

        Returns:
            DataLoader yielding Batch objects with numpy arrays
        """
        from torch.utils.data import DataLoader

        dataset = ray.get(
            self._dataset_manager.get_val_batch_dataset_view.remote(keys, filter=None)
        )
        collate_fn = batch_collate_fn(self._dataset_manager)
        return DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, **kwargs)

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
    dump_config=None,
):
    dataset_manager_actor = DatasetManagerActor.remote(
        min_training_samples=min_training_samples,
        max_training_samples=max_training_samples,
        validation_window_size=validation_window_size,
        resample_batch_size=resample_batch_size,
        resample_interval=resample_interval,
        keep_resampling=keep_resampling,
        initial_samples_path=initial_samples_path,
        dump_config=dump_config,
    )
    dataset_manager = DatasetManager(dataset_manager_actor)
    return dataset_manager
