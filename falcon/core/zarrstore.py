import ray
import zarr
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
    def __init__(self, sim_dir: Path, shapes_and_dtypes, 
                 num_max_sims = None,   # TODO: Maximum number of simulations to store
                 num_min_sims = None,   # TODO: Minimum number of simulations to train on
                 num_val_sims = None,   # TODO: Number of sliding validation sims
                 num_resims = 256,
                 ):
        self.num_max_sims = num_max_sims
        self.num_val_sims = num_val_sims
        self.num_min_sims = num_min_sims
        self.num_resims = num_resims
        self.length = 0

        # Initialize the Zarr store with given shapes
        self.zarr_store_path = sim_dir.expanduser().resolve() / "dataset.zarr"
        self.shapes_and_dtypes = shapes_and_dtypes
        self.store = zarr.open_group(self.zarr_store_path, mode='a')
        self.arrays = {}

        for key, shape_and_dtype in shapes_and_dtypes.items():
            if key not in self.store:
                shape, dtype = shape_and_dtype
                self.arrays[key] = self.store.create_dataset(
                    key,
                    shape=(0,) + shape,
                    chunks=(CHUNK_SIZE,) + shape,
                    dtype=dtype,
                    overwrite=True,
                    append_dim=0,
                    #compressor=None
                )
            else:
                self.arrays[key] = self.store[key]

        # Create active column with boolen values
        if 'active' not in self.store:
            self.arrays['active'] = self.store.create_dataset(
                'active',
                shape=(0,),
                chunks=(1024,),
                dtype='bool',
                overwrite=True,
                append_dim=0
            )
        else:
            self.arrays['active'] = self.store['active']

        initialize_logging_for("Dataset")    # Set logger for global scope of this actor

        asyncio.create_task(self.monitor())

    async def monitor(self):
        while True:
            #print("🖥️ Monitor step")
            self._update_length()
            self._deactivate_excess_samples(self.num_max_sims)
            await asyncio.sleep(1.0)

    def _update_length(self):
        """Update the length of the dataset based on the minimum length of all arrays."""
        self.length = min(len(self.arrays[key]) for key in self.arrays.keys())
        #print([f"{key}: {len(self.arrays[key])}" for key in self.arrays.keys()])
        #print("Dataset length:", self.length)
        log({"Dataset:length": self.length})

    def _deactivate_excess_samples(self, max_active_samples):
        """Deactivate samples that are older than num_min_sims + num_val_sims."""
        active_ids = np.where(self.arrays['active'][:])[0]
        if len(active_ids) > max_active_samples:
            self.deactivate(active_ids[:len(active_ids) - max_active_samples])

    def get_num_min_sims(self):
        return self.num_min_sims

    def get_num_resims(self):
        return self.num_resims

    def append(self, data):
        """Append data directly to the end of all arrays.
        
        This is the original append behavior that simply adds data to the end
        of the arrays without considering inactive slots.
        
        Args:
            data: Dictionary mapping array keys to new data arrays
        """
        # Append data to the datasets
        for key, value in data.items():
            #print(f"Appending to key: {key}, value shape: {value.shape}")
            if not key in self.arrays.keys():
                continue
            try:
                self.arrays[key].append(value)
            except ValueError as e:
                array_shape = self.arrays[key].shape  # Get the current shape of the array in the store
                value_shape = value.shape  # Get the shape of the value being appended
                raise ValueError(
                    f"Error appending to array with key '{key}'. "
                    f"Original shape: {array_shape}, appended shape: {value_shape}. Original error: {e}"
            ) from e
        # Extend active column accordingly
        self.arrays['active'].append(np.ones(value.shape[0], dtype=bool))

    def update(self, data):
        self.append(data)

    def get_length(self):
        # Return minimum length of all datasets
        return self.length

    def get_num_active(self):
        # Return number of active samples
        return np.sum(self.arrays['active'][:self.get_length()])

    def is_active(self):
        return self.arrays['active'][:self.get_length()]

    def get_active_ids(self):
        # Get indices of active samples
        return np.where(self.arrays['active'][:self.get_length()])[0]

    def deactivate(self, ids):
        # Deactivate samples by setting active column to False
        if len(ids) > 0:
            #print("🧨 Deactivating samples (min, max, len):", min(ids), max(ids), len(ids))
            self.arrays['active'][ids] = False
            for key in self.arrays.keys():
                if key != 'active':
                    # self.arrays[key][ids] = 0  # Very very slow
                    for i in ids:
                        self.arrays[key][i] = 0
            #print("🧨 ...done")

#    def get_dataset_views(self, keys, filter_train=None, filter_val=None, active_only=False):
#        slice_train = [-self.num_min_sims-1,-self.num_val_sims-1]
#        slice_val = [-self.num_val_sims-1,-1]
#        dataset_train = DatasetView(self.filepath, keys, filter=filter_train,
#            active_only=active_only, slice=slice_train)
#        dataset_val = DatasetView(self.filepath, keys, filter=filter_val,
#            active_only=active_only, slice=slice_val)
#        return dataset_train, dataset_val

    def get_train_dataset_view(self, keys, filter=None, active_only=False):
        slice = [-self.num_min_sims-1,-self.num_val_sims-1]
        dataset = DatasetView(self.zarr_store_path, keys, filter=filter,
            active_only=active_only, slice=slice)
        return dataset

    def get_val_dataset_view(self, keys, filter=None, active_only=False):
        slice = [-self.num_val_sims-1,-1]
        dataset_train = DatasetView(self.zarr_store_path, keys, filter=filter,
            active_only=active_only, slice=slice)
        return dataset_train

    def generate_samples(self, deployed_graph, num_sims):
        samples = deployed_graph.sample(num_sims)
        self.append(samples)

    def shutdown(self):
        pass
        #shutdown_global_logger()

#class DatasetView(IterableDataset):
#    def __init__(self, zarr_store_path, keylist, filter=None, slice=None, active_only=False, pre_load=True, shuffle=True):
#        self.dataset_manager = ray.get_actor("DatasetManager")
#        self.keylist = keylist
#        self.store = zarr.open_group(zarr_store_path, mode='r')
#        self.filter = filter
#        self.slice = slice
#        self.active_only = active_only
#        self.pre_load = pre_load
#        self.shuffle = shuffle
#        self.cache = None
#        self.cache_ids = None
#
#    def __iter__(self):
#        if self.active_only:
#            ids = ray.get(self.dataset_manager.get_active_ids.remote())
#        else:
#            length = ray.get(self.dataset_manager.get_length.remote())
#            ids = list(range(0, length))
#        if self.slice is not None:
#            ids = ids[self.slice[0]:self.slice[1]]
#        if self.shuffle:
#            random.shuffle(ids)
#        if self.pre_load:
#            time_start = time.time()
#            if self.cache is None:
#                self.cache = {k: self.store[k][ids] for k in self.keylist}
#                self.cache_ids = ids
#            time_end = time.time()
#            log({"Dataset:pre_load_time [sec]": time_end - time_start})
#        for i, index in enumerate(self.cache_ids):
#            if self.pre_load:
#                sample = [self.cache[key][i] for key in self.keylist]
#            else:
#                sample = [self.store[key][index] for key in self.keylist]
#            if self.filter is not None:  # Online evaluation
#                sample = self.filter(sample)
#            yield (index, *sample)

class ZarrStore:
    def __init__(self, zarr_store_path):
        self.store = zarr.open_group(zarr_store_path, mode='r')

    def get_active_sample_ids(self) -> np.ndarray:
        # All of the present once are assumed to be active
        return self.store['sample_id'][:]

    def get_item(self, id, key):
        """Get item by id and key."""
#        sample_ids = self.store['sample_id'][:]
#        if id not in sample_ids:
#            raise KeyError(f"Sample ID {id} not found in active samples.")
#        index = np.where(sample_ids == id)[0][0]
        return self.store[key][id][:]

class DatasetView(IterableDataset):
    def __init__(self, zarr_store_path, keylist, filter=None, slice=None, active_only=False, pre_load=True, shuffle=True):
        self.dataset_manager = ray.get_actor("DatasetManager")
        #self.store = zarr.open_group(zarr_store_path, mode='r')
        self.store = ZarrStore(zarr_store_path)
        self.keylist = keylist
        self.filter = filter
        self.slice = slice
        self.cache = {}  # structure is {id1: {key1: array1, key2: array2, ...}, id2: {...}, ...}

    def update_cache(self):
        active_sample_ids = ray.get(self.dataset_manager.get_active_ids.remote())  # Ids as key list

        # Based on slice select most recent ids (for validation or training etc)
        if self.slice is not None:
            active_sample_ids = active_sample_ids[self.slice[0]:self.slice[1]]


        # Remove old ids from cache
        remove_ids = list(set(self.cache.keys()) - set(active_sample_ids))
        for remove_id in remove_ids:
            del self.cache[remove_id]

        # New ids that are not in self.cache_ids yet
        new_ids = list(set(active_sample_ids) - set(self.cache.keys()))
        for new_id in new_ids:
            self.cache[new_id] = {key: self.store.get_item(new_id, key) for key in self.keylist}

    def __iter__(self):
        time_start = time.time()
        self.update_cache()
        time_end = time.time()
        log({"Dataset:cache_update_time [sec]": time_end - time_start})

        # Shuffle indices for iteration
        indices = list(self.cache.keys())
        random.shuffle(indices)

        # Iterate over the cache
        for index in indices:
            sample = [self.cache[index][key] for key in self.keylist]
            if self.filter is not None:  # Online evaluation
                sample = self.filter(sample)
            yield (index, *sample)

def get_zarr_dataset_manager(shapes_and_dtypes, sim_dir: Path, num_min_sims=None,
                             num_val_sims=None, num_resims=64, num_max_sims=None):
    #shapes_and_dtypes = deployed_graph.get_shapes_and_dtypes()
    assert isinstance(sim_dir, Path), "sim_dir must be a Path object"
    dataset_manager_actor = DatasetManagerActor.remote(sim_dir, shapes_and_dtypes,
            num_min_sims=num_min_sims,
            num_max_sims=num_max_sims,
            num_val_sims=num_val_sims,
            num_resims=num_resims
    )
    dataset_manager = DatasetManager(dataset_manager_actor)
    return dataset_manager
