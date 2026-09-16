"""Base class for estimators whose networks are torch modules."""

import copy
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn

from falcon.core.base_estimator import BaseEstimator, StateTree
from falcon.core.logger import debug


@dataclass
class NetworkGroup:
    """Modules that are snapshotted, compared and promoted together.

    Attributes:
        modules: Named modules whose state makes up the group.
        metric: Validation metric that decides whether the group is promoted.
        embedding: Embedding feeding the group, if any (also one of ``modules``).
    """

    modules: Dict[str, nn.Module]
    metric: str
    embedding: Optional[nn.Module] = None


def to_numpy_tree(value: Any) -> Any:
    """Copy tensors in a (nested) state dict to numpy; plain values pass through."""
    if isinstance(value, torch.Tensor):
        value = value.detach().to("cpu", copy=True)
        if value.dtype == torch.bfloat16:
            value = value.float()
        return value.numpy()
    if isinstance(value, dict):
        return {k: to_numpy_tree(v) for k, v in value.items()}
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(
        f"Cannot export a {type(value).__name__} from a module state; "
        "keep tensors in registered buffers or parameters"
    )


def to_tensor_tree(value: Any, device) -> Any:
    """Inverse of ``to_numpy_tree``: arrays become fresh tensors on ``device``."""
    if isinstance(value, np.ndarray):
        # Arrays received through Ray are read-only views; always copy
        return torch.from_numpy(np.array(value, copy=True)).to(device)
    if isinstance(value, dict):
        return {k: to_tensor_tree(v, device) for k, v in value.items()}
    return value


def module_state(module: nn.Module) -> StateTree:
    return to_numpy_tree(dict(module.state_dict()))


def load_module_state(module: nn.Module, tree: StateTree, device) -> None:
    module.load_state_dict(to_tensor_tree(tree, device))


class TorchModel(BaseEstimator):
    """
    Implements the state and sampling plumbing of ``BaseEstimator`` for torch.

    Subclasses register their networks with ``_set_groups()`` in ``build()``
    and implement:

    - ``init_from_batch()`` / ``build()``
    - ``train_step()`` / ``val_step()`` / ``discard_test()``
    - ``_sample_prior()`` / ``_sample()``

    ``val_step``, ``discard_test`` and ``_sample`` run without gradients, and
    sampling runs with torch's random state seeded from the given ``rng``.
    """

    def setup(self, simulator_instance, theta_key, condition_keys):
        super().setup(simulator_instance, theta_key, condition_keys)
        if getattr(self, "device", None):
            self.device = torch.device(self.device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            debug(f"Auto-detected device: {self.device}")
        self._groups: Dict[str, NetworkGroup] = {}

    def set_num_threads(self, num_threads: int) -> None:
        torch.set_num_threads(num_threads)

    # ==================== Networks ====================

    def _set_groups(self, groups: Dict[str, NetworkGroup]) -> None:
        self._groups = dict(groups)

    def groups(self) -> Dict[str, str]:
        return {name: g.metric for name, g in self._groups.items()}

    @property
    def built(self) -> bool:
        return bool(self._groups)

    @property
    def cache_device(self) -> Optional[str]:
        return str(self.device) if getattr(self, "cache_on_device", False) else None

    def snapshot(self, group: str):
        return {k: copy.deepcopy(m.state_dict()) for k, m in self._groups[group].modules.items()}

    def restore(self, group: str, handle) -> None:
        for k, m in self._groups[group].modules.items():
            m.load_state_dict(handle[k])

    def export_state(self, group: str) -> StateTree:
        return {k: module_state(m) for k, m in self._groups[group].modules.items()}

    def import_state(self, group: str, tree: StateTree) -> None:
        modules = self._groups[group].modules
        if set(tree) != set(modules):
            raise KeyError(f"State of group '{group}' has modules {sorted(tree)}, expected {sorted(modules)}")
        for k, m in modules.items():
            load_module_state(m, tree[k], self.device)

    # ==================== Utilities ====================

    @staticmethod
    def _to_tensor(x, device=None):
        """Convert numpy array or torch tensor to the target device."""
        if not isinstance(x, torch.Tensor):
            x = torch.from_numpy(np.array(x, copy=True))
        return x if device is None else x.to(device)

    @staticmethod
    def _to_array(x) -> np.ndarray:
        return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.array(x, copy=True)

    def _summary(self, group: str, conditions, train: bool = False):
        """Embedding output for ``conditions``, using the embedding of a network group.

        Training and validation route all embedding calls through here.
        """
        embedding = self._groups[group].embedding
        embedding.train(train)
        return embedding(conditions)

    @contextmanager
    def _seeded(self, rng: Optional[np.random.Generator]):
        """Seed torch's random state from ``rng`` for the duration of the block."""
        if rng is None:
            yield
            return
        seed = int(rng.integers(2**63 - 1))
        devices = [self.device.index or 0] if self.device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            yield

    # ==================== Contract ====================

    def evaluate(self, batch) -> Dict[str, float]:
        with torch.no_grad():
            return self.val_step(batch)

    def discard_mask(self, batch) -> np.ndarray:
        with torch.no_grad():
            return self._to_array(self.discard_test(batch)).astype(bool)

    def sample_prior(self, rng, num_samples: int) -> dict:
        with self._seeded(rng):
            return self._sample_prior(num_samples)

    def sample(self, rng, num_samples: int, conditions, mode: str) -> dict:
        with self._seeded(rng), torch.no_grad():
            return self._sample(num_samples, conditions, mode)

    def val_step(self, batch) -> Dict[str, float]:
        raise NotImplementedError

    def discard_test(self, batch):
        """Boolean mask (tensor or array) of length len(batch); True = discard."""
        raise NotImplementedError

    def _sample_prior(self, num_samples: int) -> dict:
        raise NotImplementedError

    def _sample(self, num_samples: int, conditions, mode: str) -> dict:
        raise NotImplementedError

    # ==================== Legacy checkpoints ====================

    @staticmethod
    def _legacy_load(path: Path):
        return torch.load(path, map_location="cpu", weights_only=False)

    def _legacy_meta(self, node_dir: Path) -> Dict[str, Any]:
        meta = {"round": 0, "rounds_accepted": 0, "stall": 0, "total_epochs": 0, "has_best": True}
        if (node_dir / "round_state.pth").exists():
            state = self._legacy_load(node_dir / "round_state.pth")
            meta.update(round=state["rounds"], rounds_accepted=state["accepted"], stall=state.get("stall", 0))
        if (node_dir / "total_epochs_trained.pth").exists():
            meta["total_epochs"] = int(self._legacy_load(node_dir / "total_epochs_trained.pth"))
        return meta
