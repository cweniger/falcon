"""Abstract base class for inference estimators."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Optional

import torch

# Type alias for conditions: maps node names to tensors
Conditions = Dict[str, torch.Tensor]


class BaseEstimator(ABC):
    """
    Fully abstract base class defining the estimator interface.

    Subclasses implement a two-phase lifecycle:

    1. ``__init__``: pure config storage — stores all parameters as instance
       attributes.  Defaults live here; nothing runtime is wired up.

    2. ``setup()``: load-bearing runtime wiring — called by ``NodeWrapper``
       inside a Ray actor before training begins.  Applies any flat YAML
       overrides (``config`` dict), then initialises networks, devices, and
       all runtime objects.

    Example::

        graph.add_node("z", estimator=Flow(max_epochs=300, net_type="nsf"))
    """

    @abstractmethod
    def setup(
        self,
        simulator_instance,
        theta_key: Optional[str],
        condition_keys,
    ) -> None:
        """Wire up runtime components.

        Called by ``NodeWrapper`` before training.

        Args:
            simulator_instance: Live prior/simulator, already constructed.
            theta_key: Name of the parameter node being estimated.
            condition_keys: List of evidence/scaffold node names.
        """

    @abstractmethod
    async def train(self, buffer) -> None:
        pass

    @abstractmethod
    def sample_prior(self, num_samples: int, conditions: Optional[Conditions] = None) -> dict:
        pass

    @abstractmethod
    def sample_posterior(self, num_samples: int, conditions: Optional[Conditions] = None) -> dict:
        pass

    @abstractmethod
    def sample_proposal(self, num_samples: int, conditions: Optional[Conditions] = None) -> dict:
        pass

    @abstractmethod
    def save(self, node_dir: Path) -> None:
        pass

    @abstractmethod
    def load(self, node_dir: Path) -> None:
        pass

    @abstractmethod
    def interrupt(self) -> None:
        pass
