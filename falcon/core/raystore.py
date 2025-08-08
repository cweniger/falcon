import ray
import numpy as np
from torch.utils.data import IterableDataset
#import torch
import random
from pathlib import Path
import time
import asyncio

from falcon.core.logging import initialize_logging_for, log

CHUNK_SIZE = 1

class DatasetManager:
    """Access the DatasetManagerActor without exposing the actor interface directly."""
    def __init__(self, dataset_manager_actor):
        self.dataset_manager_actor = dataset_manager_actor

    def generate_samples(self, deployed_graph, num_sims):
        ray.get(self.dataset_manager_actor.generate_samples.remote(deployed_graph, num_sims = num_sims))

@ray.remote(name='DatasetManager')
class DatasetManagerActor:
    def __init__(self, 
                 num_max_sims = None,   # TODO: Maximum number of simulations to store
                 num_min_sims = None,   # TODO: Minimum number of simulations to train on
                 num_val_sims = None,   # TODO: Number of sliding validation sims
                 num_resims = 256,
                 ):
        self.num_max_sims = num_max_sims
        self.num_val_sims = num_val_sims
        self.num_min_sims = num_min_sims
        self.num_resims = num_resims

        # Store
        self.ray_store = []
        self.is_active = np.zeros(0, dtype=bool)
        self.ref_counts = np.zeros(0)

        initialize_logging_for("Dataset")

        asyncio.create_task(self.monitor())

    async def monitor(self):
        while True:
            #print("🖥️ Monitor step")
            self._deactivate_excess_samples(self.num_max_sims)
            await asyncio.sleep(3.0)

    def get_active_ids(self):
        return np.where(self.is_active)[0]

    def _deactivate_excess_samples(self, max_active_samples):
        """Deactivate samples that are older than num_min_sims + num_val_sims."""
        active_ids = self.get_active_ids()
        if len(active_ids) > max_active_samples:
            self.deactivate(active_ids[:len(active_ids) - max_active_samples])

    def get_num_min_sims(self):
        return self.num_min_sims

    def get_num_resims(self):
        return self.num_resims

    def get_active_samples(self):
        active_ids = self.get_active_ids()
        active_ray_store = [self.ray_store[i] for i in active_ids]
        self.ref_counts[active_ids] += 1
        return active_ray_store, active_ids

    def release_samples(self, ids):
        self.ref_counts[ids] -= 1

    def append(self, data):
        num_new_samples = data[list(data.keys())[0]].shape[0]
        for i in range(num_new_samples):
            sample = {key: ray.put(value[i]) for key, value in data.items()}
            self.ray_store.append(sample)
        self.is_active = np.append(self.is_active, np.ones(num_new_samples, dtype=bool))
        self.ref_counts = np.append(self.ref_counts, np.zeros(num_new_samples))
        log({"Dataset:length": len(self.ray_store)})
        log({"Dataset:is_active": sum(self.is_active)})

    def update(self, data):
        self.append(data)

    def get_length(self):
        return len(self.ray_store)

    def get_num_active(self):
        return np.sum(self.is_active)

    def deactivate(self, ids):
        ids = ids[self.ref_counts[ids] <= 0]  # Only deactivate samples that are not referenced anymore
        for i in ids:
            self.is_active[i] = False
            ray.internal.free(list(self.ray_store[i].values()))  # Remove all traces of the sample
            self.ray_store[i] = None  

    def get_train_dataset_view(self, keys, filter=None, active_only=False):
        slice = [-self.num_min_sims-1,-self.num_val_sims-1]
        dataset = DatasetView(keys, filter=filter,
            active_only=active_only, slice=slice)
        return dataset

    def get_val_dataset_view(self, keys, filter=None, active_only=False):
        slice = [-self.num_val_sims-1,-1]
        dataset_train = DatasetView(keys, filter=filter,
            active_only=active_only, slice=slice)
        return dataset_train

    def generate_samples(self, deployed_graph, num_sims):
        samples = deployed_graph.sample(num_sims)
        self.append(samples)

    def shutdown(self):
        pass
        #shutdown_global_logger()


class DatasetView(IterableDataset):
    def __init__(self, keylist, filter=None, slice=None, active_only=False, pre_load=True, shuffle=True):
        self.dataset_manager = ray.get_actor("DatasetManager")
        self.keylist = keylist
        self.filter = filter
        self.slice = slice

    def __iter__(self):
        active_samples, all_active_ids = ray.get(self.dataset_manager.get_active_samples.remote())

        if self.slice is not None:
            active_samples = active_samples[self.slice[0]:self.slice[1]]
            active_ids = all_active_ids[self.slice[0]:self.slice[1]]

        perm = np.random.permutation(len(active_samples))

        log({"DatasetView:length": len(perm)})

        print("Starting iterating", len(perm))
        for i in perm:
            sample = [ray.get(active_samples[i][key]) for key in self.keylist]
            if self.filter is not None:  # Online evaluation
                sample = self.filter(sample)
            index = active_ids[i]
            yield (index, *sample)
        print("Done iterating", len(perm))

        self.dataset_manager.release_samples.remote(all_active_ids)

def get_ray_dataset_manager(num_min_sims=None, num_val_sims=None, num_resims=64, num_max_sims=None):
    #shapes_and_dtypes = deployed_graph.get_shapes_and_dtypes()
    dataset_manager_actor = DatasetManagerActor.remote(
            num_min_sims=num_min_sims,
            num_max_sims=num_max_sims,
            num_val_sims=num_val_sims,
            num_resims=num_resims
    )
    dataset_manager = DatasetManager(dataset_manager_actor)
    return dataset_manager
