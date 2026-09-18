"""Framework-neutral state trees and the checkpoint files built from them.

A state tree is a nested dict with string keys whose leaves are numpy arrays or
plain JSON values (numbers, strings, booleans, None, and lists of those). It is
what the train actor sends to the sample actors and what a checkpoint holds, so
neither side needs to know which framework the networks use.

On disk a tree is one ``.npz`` file: arrays under their ``/``-joined key path,
everything else in a JSON entry.
"""

import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np

CHECKPOINT_NAME = "best_state.npz"
_JSON_KEY = "__json__"
_SEP = "/"


def _flatten(tree: Dict[str, Any], prefix: str, arrays: dict, values: dict) -> None:
    if not tree and prefix:
        values[prefix.rstrip(_SEP)] = {}
    for key, value in tree.items():
        if not isinstance(key, str) or _SEP in key or not key:
            raise ValueError(f"State tree keys must be non-empty strings without '{_SEP}': {key!r}")
        path = prefix + key
        if isinstance(value, dict):
            _flatten(value, path + _SEP, arrays, values)
        elif isinstance(value, np.ndarray):
            arrays[path] = value
        elif isinstance(value, np.generic):
            values[path] = value.item()
        else:
            try:
                json.dumps(value)
            except TypeError as e:
                raise TypeError(
                    f"State tree leaf '{path}' is a {type(value).__name__}; "
                    "leaves must be numpy arrays or plain JSON values"
                ) from e
            values[path] = value


def _insert(tree: dict, path: str, value: Any) -> None:
    *parents, leaf = path.split(_SEP)
    node = tree
    for part in parents:
        node = node.setdefault(part, {})
    node[leaf] = value


def save_state(path, tree: Dict[str, Any]) -> None:
    """Write a state tree to ``path`` atomically."""
    arrays, values = {}, {}
    _flatten(tree, "", arrays, values)
    if _JSON_KEY in arrays:
        raise ValueError(f"'{_JSON_KEY}' is reserved")
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.savez(f, **arrays, **{_JSON_KEY: np.array(json.dumps(values))})
    os.replace(tmp, path)


def load_state(path) -> Dict[str, Any]:
    """Read a state tree written by ``save_state``."""
    tree: Dict[str, Any] = {}
    with np.load(path, allow_pickle=False) as data:
        for key in data.files:
            if key != _JSON_KEY:
                _insert(tree, key, data[key])
        for key, value in json.loads(data[_JSON_KEY].item()).items():
            _insert(tree, key, value)
    return tree


def write_checkpoint(node_dir, tree: Dict[str, Any]) -> Path:
    node_dir = Path(node_dir)
    node_dir.mkdir(parents=True, exist_ok=True)
    path = node_dir / CHECKPOINT_NAME
    save_state(path, tree)
    return path


def read_checkpoint(node_dir, legacy: Optional[Callable] = None) -> Optional[Dict[str, Any]]:
    """Checkpoint of a node, or None if there is none.

    Args:
        node_dir: The node's directory, e.g. ``graph/z``.
        legacy: Optional callable(node_dir) -> tree or None that reads checkpoints
            written before ``best_state.npz`` existed.
    """
    node_dir = Path(node_dir)
    path = node_dir / CHECKPOINT_NAME
    if path.exists():
        return load_state(path)
    if legacy is not None:
        return legacy(node_dir)
    return None


def has_checkpoint(node_dir) -> bool:
    node_dir = Path(node_dir)
    return (node_dir / CHECKPOINT_NAME).exists() or any(node_dir.glob("*.pth"))
