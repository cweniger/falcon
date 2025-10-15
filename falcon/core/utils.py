import importlib
import os
import torch
import numpy as np


class RV:
    """
    Random Variable object that encapsulates:
      - value(s) produced at a node
      - log probability (if available)
      - optional metadata
    """

    def __init__(self, value, logprob=None, name=None, meta=None):
        self.value = value  # tensor, array, or dict of values
        self.logprob = logprob  # scalar or dict of logprobs
        self.name = name  # optional identifier
        self.meta = meta or {}  # arbitrary metadata (dict)

    def __repr__(self):
        return (
            f"RV(name={self.name}, " f"value={self.value}, " f"logprob={self.logprob})"
        )

    def to_dict(self):
        """Serialize to dict form for storage or logging."""
        return {
            "name": self.name,
            "value": self.value,
            "logprob": self.logprob,
            "meta": self.meta,
        }


def as_rv(value, logprob=None, name=None, meta=None):
    """
    Ensure output is an RV. If already RV, return unchanged; otherwise wrap it.
    """
    if isinstance(value, RV):
        return value
    return RV(value=value, logprob=logprob, name=name, meta=meta)


class RVBatch:
    """
    Batched Random Variable container that encapsulates:
      - value(s) with leading batch dimension
      - log probability per item (if available)
      - optional metadata
      - optional per-item names
    """

    def __init__(self, value, logprob=None, names=None, meta=None):
        self.value = value  # tensor/array or dict of tensors, shape [B, ...]
        self.logprob = (
            logprob  # tensor/array or dict aligned with value, shape [B, ...] or [B]
        )
        self.names = names  # optional sequence of length B
        self.meta = meta or {}  # arbitrary metadata (dict)

        self.batch_size = self._infer_batch_size()

    def _infer_batch_size(self):
        if self.value is None:
            return 0
        if isinstance(self.value, dict):
            # take first field to infer
            for v in self.value.values():
                return v.shape[0]
            return 0
        return self.value.shape[0]

    def __repr__(self):
        return (
            f"RVBatch(B={self.batch_size}, "
            f"value={self.value}, "
            f"logprob={self.logprob}, "
            f"names={self.names}, "
            f"meta_keys={list(self.meta)})"
        )

    def to_dict(self):
        """Serialize to dict form for storage or logging."""
        return {
            "value": self.value,
            "logprob": self.logprob,
            "names": self.names,
            "meta": self.meta,
        }


def as_rvbatch(value, logprob=None, names=None, meta=None):
    """
    Ensure output is an RVBatch. If `value` is already an RVBatch,
    return it unchanged; otherwise wrap it.
    """
    if isinstance(value, RVBatch):
        return value
    return RVBatch(value=value, logprob=logprob, names=names, meta=meta)


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
