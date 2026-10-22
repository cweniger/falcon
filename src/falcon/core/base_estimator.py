"""Model contract shared by all estimators, independent of the network framework."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# Maps node names to arrays in the model's own framework
Conditions = Dict[str, Any]
# Nested dict of numpy arrays and plain values (see falcon.core.state_io)
StateTree = Dict[str, Any]


@dataclass
class RoundConfig:
    """Parameters of the round-based training loop (see docs/training.md)."""

    max_epochs: int = 100
    patience_epochs: int = 16
    val_every_epochs: int = 1
    max_rounds: Optional[int] = None
    patience_rounds: int = 10
    prior_rounds: int = 0
    batch_size: int = 128
    max_cache_samples: int = 0
    cache_on_device: bool = False
    discard_samples: bool = False


class BaseEstimator(ABC):
    """
    Model of one estimator family, used unchanged by both actors of a node.

    The train actor (``<node>/train``) drives it through ``RoundTrainer``, the
    sample actor (``<node>``) through ``ModelSampler``. The model holds its
    networks as mutable state on ``self`` and knows nothing about rounds,
    publishing or Ray; each actor calls only the methods it needs.

    Lifecycle:

    1. ``__init__``: pure config storage (flat YAML kwargs). The round-loop
       parameters are read from attributes named like the ``RoundConfig``
       fields.
    2. ``setup()``: runtime wiring inside the actor (simulator, keys, device).
    3. ``build(init_tree)``: creates the networks. The trainer gets the tree
       from ``init_from_batch()`` on its first batch; the sampler gets it from
       the published state.

    State rules, so that any framework can implement this:

    - ``snapshot()`` / ``restore()`` keep and restore fast in-memory copies of
      a network group; the handles are opaque.
    - ``export_state()`` returns copies as numpy trees; ``import_state()``
      installs such a tree. Nothing outside the model keeps references to the
      model's own arrays.
    - Randomness in sampling comes from the ``rng`` argument.

    Example::

        graph.add_node("z", estimator=Flow(max_epochs=300, net_type="nsf"))
    """

    # ==================== Setup ====================

    def setup(self, simulator_instance, theta_key: str, condition_keys: List[str]) -> None:
        """Wire up runtime components inside the actor.

        Args:
            simulator_instance: Live prior/simulator of the node.
            theta_key: Name of the node being estimated.
            condition_keys: Evidence and scaffold node names.
        """
        self.simulator_instance = simulator_instance
        self.theta_key = theta_key
        self.condition_keys = list(condition_keys or [])

    @property
    def round_config(self) -> RoundConfig:
        defaults = RoundConfig()
        return RoundConfig(**{
            f.name: getattr(self, f.name, getattr(defaults, f.name)) for f in fields(RoundConfig)
        })

    @property
    def cache_device(self) -> Optional[str]:
        """Device for the training data cache (None = CPU)."""
        return None

    def set_num_threads(self, num_threads: int) -> None:
        """Limit the CPU threads of the network framework in this process.

        Called by the actor, whose CPU budget is shared with the node's other
        actor. Frameworks that honour ``OMP_NUM_THREADS`` at import need
        nothing here.
        """

    def batch_keys(self) -> List[str]:
        """Buffer keys the training batches must contain."""
        return [f"{self.theta_key}.value", f"{self.theta_key}.log_prob",
                *[f"{k}.value" for k in self.condition_keys]]

    @property
    def history(self) -> Dict[str, list]:
        """Model-specific training history, saved next to the loop's history."""
        return {}

    # ==================== Networks ====================

    @abstractmethod
    def groups(self) -> Dict[str, str]:
        """Network groups mapped to their validation metric; the first is primary.

        Groups are compared and promoted independently; a round is accepted
        when the primary group improves.
        """

    @property
    @abstractmethod
    def built(self) -> bool:
        """Whether ``build()`` has been called."""

    @abstractmethod
    def init_from_batch(self, batch) -> StateTree:
        """What ``build()`` needs, taken from the first training batch."""

    @abstractmethod
    def build(self, init_tree: StateTree) -> None:
        """Create the networks."""

    @abstractmethod
    def snapshot(self, group: str) -> Any:
        """Opaque in-memory copy of a group's current state."""

    @abstractmethod
    def restore(self, group: str, handle: Any) -> None:
        """Restore a group from ``snapshot()``."""

    @abstractmethod
    def export_state(self, group: str) -> StateTree:
        """Copy of a group's state as a numpy tree."""

    @abstractmethod
    def import_state(self, group: str, tree: StateTree) -> None:
        """Install a group's state from a numpy tree."""

    def load_legacy(self, node_dir: Path) -> Optional[StateTree]:
        """Read a checkpoint written before ``best_state.npz`` existed.

        Returns a tree in the published format, or None.
        """
        return None

    # ==================== Training ====================

    @abstractmethod
    def train_step(self, batch) -> Dict[str, float]:
        """One optimizer update on one training batch; returns metrics incl. "loss"."""

    @abstractmethod
    def evaluate(self, batch) -> Dict[str, float]:
        """Validation metrics for one batch, without updating anything.

        Must include "loss" (early stopping) and the metric of every group
        (acceptance test).
        """

    @abstractmethod
    def discard_mask(self, batch) -> np.ndarray:
        """Discard test for one batch with the primary group; True = discard.

        Called after an accepted round, when the primary group equals the
        best network.
        """

    def on_round_start(self) -> None:
        """Called at the start of every round after the first, once the best
        state has been restored (e.g. reset the learning rate)."""

    def on_validation_end(self, epoch: int, val_metrics: Dict[str, float]) -> Optional[Dict[str, float]]:
        """Called after each validation within a round (e.g. LR scheduler step).

        Returns:
            Optional dict of extra metrics for the epoch summary line.
        """
        return None

    # ==================== Sampling ====================

    @abstractmethod
    def sample_prior(self, rng: np.random.Generator, num_samples: int) -> dict:
        """Prior samples: ``{"value": ndarray, "log_prob": ndarray}``."""

    @abstractmethod
    def sample(self, rng: np.random.Generator, num_samples: int,
               conditions: Conditions, mode: str) -> dict:
        """Samples from the networks: ``{"value": ndarray, "log_prob": ndarray}``.

        Args:
            rng: Source of randomness for this call.
            num_samples: Number of samples.
            conditions: Condition arrays (numpy); a leading dimension of 1 is
                broadcast over all samples.
            mode: ``"proposal"`` (tempered, used for simulations) or ``"posterior"``.
        """
