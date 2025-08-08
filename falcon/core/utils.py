import importlib
import os
import torch
import numpy as np

class LazyLoader:
    """This class is used to lazily load a class from a string path."""
    def __init__(self, class_path, *args, **kwargs):
        self.class_path = class_path
        self.args = args
        self.kwargs = kwargs

    def __getstate__(self):
        return {"class_path": self.class_path, "args": self.args, "kwargs": self.kwargs}
    
    def __setstate__(self, state):
        self.class_path = state["class_path"]
        self.args = state["args"]
        self.kwargs = state["kwargs"]
    
    def __call__(self, **kwargs):
        local_config = {**self.kwargs, **kwargs}
        module_name, class_name = self.class_path.rsplit(".", 1)
        module = importlib.import_module(module_name)
        model_class = getattr(module, class_name)
        model_instance = model_class(*self.args, **local_config)
        return model_instance


def load_observations(config):
    """Load observations from NPZ file with optional batch indexing.
    
    Args:
        config: Dict with 'path' (required) and 'index' (optional)
                - {"path": "data/observations.npz"}
                - {"path": "data/observations.npz", "index": 0}
                - {"path": "data/observations.npz", "index": null}
        
    Returns:
        dict: Dictionary of observations with node names as keys
    """
    data_path = config['path']
    index = config.get('index')
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Observation file not found: {data_path}")
    
    # Load from NPZ file
    data = np.load(data_path)
    observations = {}
    
    for key in data.files:
        array = data[key]
        if index is not None:
            # Select specific batch index, removes batch dimension
            observations[key] = torch.from_numpy(array[index])
        else:
            # Use arrays as-is
            observations[key] = torch.from_numpy(array)
    
    return observations