import numpy as np
import asyncio
from enum import IntEnum
import joblib

import ray
from torch.utils.data import IterableDataset

from falcon.core.logging import initialize_logging_for, log

class DatasetManager:
    """Access the DatasetManagerActor without exposing the actor interface directly."""
    def __init__(self, dataset_manager_actor):
        self.dataset_manager_actor = dataset_manager_actor

    def initialize_samples(self, deployed_graph):
        ray.get(self.dataset_manager_actor.initialize_samples.remote(deployed_graph))

class SampleStatus(IntEnum):
    VALIDATION = 0
    TRAINING = 1
    TOMBSTONE = 2
    DELETED = 3

@ray.remote(name='DatasetManager')
class DatasetManagerActor:
    def __init__(self, 
                 max_training_samples = None,   # TODO: Maximum number of simulations to store
                 min_training_samples = None,   # TODO: Minimum number of simulations to train on
                 validation_window_size = None,   # TODO: Number of sliding validation sims
                 resample_batch_size = 256,
                 resample_interval = 5,
                 initial_samples_path = None,
                 ):
        self.max_training_samples = max_training_samples
        self.min_training_samples = min_training_samples
        self.validation_window_size = validation_window_size
        self.resample_batch_size = resample_batch_size
        self.resample_interval = resample_interval
        self.initial_samples_path = initial_samples_path

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
        return self.resample_batch_size

    def get_resample_interval(self):
        return self.resample_interval

#    def get_active_ids(self):
#        # Only return the last num_min_sims active IDs
#        return np.where(self.status >= SampleStatus.TRAINING)[0]

#    @property
#    def training_ids(self):
#        return np.where(self.status == SampleStatus.TRAINING)[0]
#
#    @property
#    def validation_ids(self):
#        return np.where(self.status == SampleStatus.VALIDATION)[0]
#
#    @property
#    def tombstone_ids(self):
#        return np.where(self.status == SampleStatus.TOMBSTONE)[0]

    def rotate_sample_buffer(self):
        """
        Rotate samples through lifecycle: VAL -> TRAIN -> TOMBSTONE.
        
        Keeps most recent samples for validation, older samples for training,
        and marks oldest samples as tombstones for deletion.
        """
        # Get all samples that are not expired yet
        live_ids = np.where(self.status < SampleStatus.TOMBSTONE)[0]

        # Release older samples for training
        validation_ids = live_ids[:-self.validation_window_size]
        self.status[validation_ids] = SampleStatus.TRAINING

        # Set earliest samples to tombstone
        tombstone_ids = live_ids[:-self.validation_window_size - self.max_training_samples]
        self.status[tombstone_ids] = SampleStatus.TOMBSTONE

    def get_num_resims(self):
        return self.num_resims

    def get_samples_by_status(self, status):
        status_ids = np.where(self.status == status)[0]
        status_samples = [self.ray_store[i] for i in status_ids]
        self.ref_counts[status_ids] += 1
        return status_samples, status_ids

    def release_samples(self, ids):
        self.ref_counts[ids] -= 1

    def deactivate(self, ids):
        print("Deactivating samples:", ids)
        #self.status[ids[0]] = SampleStatus.TOMBSTONE

    def append(self, data, batched=True):
        if batched:  # TODO: Legacy structure
            num_new_samples = data[list(data.keys())[0]].shape[0]
            for i in range(num_new_samples):
                sample = {key: ray.put(value[i]) for key, value in data.items()}
                self.ray_store.append(sample)
        else:  # TODO: Should become default
            num_new_samples = len(data)
            for sample in data:
                sample_ray_objects = {key: ray.put(value) for key, value in sample.items()}
                self.ray_store.append(sample_ray_objects)
        self.status = np.append(self.status, np.full(num_new_samples, SampleStatus.VALIDATION))
        self.ref_counts = np.append(self.ref_counts, np.zeros(num_new_samples))

        self.rotate_sample_buffer()

        log({"Dataset:total_length": len(self.ray_store)})
        log({"Dataset:validation": sum(self.status == SampleStatus.VALIDATION)})
        log({"Dataset:training": sum(self.status == SampleStatus.TRAINING)})
        log({"Dataset:tombstone": sum(self.status == SampleStatus.TOMBSTONE)})
        log({"Dataset:deleted": sum(self.status == SampleStatus.DELETED)})

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
            #print(f"Garbage collecting {len(unreferenced_tombstone_ids)} tombstones")
            for i in unreferenced_tombstone_ids:
                ray.internal.free(list(self.ray_store[i].values()))
                self.ray_store[i] = None
                self.status[i] = SampleStatus.DELETED

    def get_train_dataset_view(self, keys, filter=None):
        dataset_train = DatasetView(keys, filter=filter, sample_status=SampleStatus.TRAINING)
        return dataset_train

    def get_val_dataset_view(self, keys, filter=None):
        dataset_val = DatasetView(keys, filter=filter, sample_status=SampleStatus.VALIDATION)
        return dataset_val

    def initialize_samples(self, deployed_graph):
        num_initial_samples = self.num_initial_samples()
        if self.initial_samples_path is not None:
            # Load initial samples from the specified path
            initial_samples = joblib.load(self.initial_samples_path)
            num_loaded_samples = len(initial_samples)
            self.append(initial_samples, batched=False)
        else:
            num_loaded_samples = 0
        if num_initial_samples > num_loaded_samples:
            samples = deployed_graph.sample(num_initial_samples - num_loaded_samples)
            self.append(samples)

    def shutdown(self):
        pass
        #shutdown_global_logger()


class DatasetView(IterableDataset):
    def __init__(self, keylist, filter=None, sample_status=SampleStatus.TRAINING):
        self.dataset_manager = ray.get_actor("DatasetManager")
        self.keylist = keylist
        self.filter = filter
        self.sample_status = sample_status

    def __iter__(self):
        active_samples, active_ids = ray.get(self.dataset_manager.get_samples_by_status.remote(self.sample_status))

        perm = np.random.permutation(len(active_samples))

        log({"DatasetView:length": len(perm)})
        #print("Starting iterating", len(perm))

        for i in perm:
            sample = [ray.get(active_samples[i][key]) for key in self.keylist]
            if self.filter is not None:  # Online evaluation
                sample = self.filter(sample)
            index = active_ids[i]
            yield (index, *sample)
        #print("Done iterating", len(perm))

        ray.get(self.dataset_manager.release_samples.remote(active_ids))

def get_ray_dataset_manager(min_training_samples=None,
        max_training_samples=None, validation_window_size=None, resample_batch_size=64,
        resample_interval=5, initial_samples_path=None
        ):
    dataset_manager_actor = DatasetManagerActor.remote(
            min_training_samples=min_training_samples,
            max_training_samples=max_training_samples,
            validation_window_size=validation_window_size,
            resample_batch_size=resample_batch_size,
            resample_interval=resample_interval,
            initial_samples_path=initial_samples_path,
    )
    dataset_manager = DatasetManager(dataset_manager_actor)
    return dataset_manager
