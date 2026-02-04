import importlib
import os
import torch
import numpy as np


class LazyLoader:
    """This class is used to lazily load a class from a string path."""

    def __init__(self, class_path):
        self.class_path = class_path

    def __call__(self, *args, **kwargs):
        if isinstance(self.class_path, str):
            module_name, class_name = self.class_path.rsplit(".", 1)
            module = importlib.import_module(module_name)
            model_class = getattr(module, class_name)
        else:
            model_class = self.class_path
        model_instance = model_class(*args, **kwargs)
        return model_instance


# TODO: Currently not used anywhere, add tests?
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
    data_path = config["path"]
    index = config.get("index")

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
