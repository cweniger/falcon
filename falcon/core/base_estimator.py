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

    Subclasses declare ``_CONFIG_SECTIONS`` (mapping section name →
    dataclass type) to get a flat keyword-argument ``__init__`` for free::

        class MyEstimator(StepwiseEstimator):
            _CONFIG_SECTIONS = {"loop": TrainingLoopConfig, "network": NetworkConfig}
            _CONFIG_EXTRA_PARAMS = [Parameter("device", KEYWORD_ONLY, default=None)]

    ``MyEstimator(loop_max_epochs=200, network_hidden_dim=64)`` then creates a
    configured instance.  :meth:`setup` is called by ``NodeWrapper`` when the
    estimator is deployed inside a Ray actor, supplying the runtime objects
    (simulator, key names).  ``__init__`` is therefore a pure dataclass —
    just storing config — while ``setup`` is load-bearing.
    """

    _CONFIG_SECTIONS: dict = {}
    _CONFIG_EXTRA_PARAMS: list = []

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if "_CONFIG_SECTIONS" not in cls.__dict__:
            return
        # Inject a class-specific __init__ with the flat signature so that
        # each subclass has its own __init__ object (not shared with the parent).
        try:
            from falcon.core.flat_config import make_flat_signature
            sig = make_flat_signature(
                cls._CONFIG_SECTIONS,
                cls.__dict__.get("_CONFIG_EXTRA_PARAMS", []),
            )

            def __init__(self, **flat_kwargs):
                BaseEstimator.__init__(self, **flat_kwargs)

            __init__.__signature__ = sig
            __init__.__qualname__ = f"{cls.__qualname__}.__init__"
            cls.__init__ = __init__
        except ImportError:
            pass

    def __init__(self, **flat_kwargs):
        from falcon.core.flat_config import flat_to_nested
        self._init_flat_kwargs = flat_to_nested(
            flat_kwargs, self.__class__._CONFIG_SECTIONS
        )

    @abstractmethod
    def setup(
        self,
        simulator_instance,
        theta_key: Optional[str],
        condition_keys,
        config=None,
    ) -> None:
        """Wire up runtime components.

        Called by ``NodeWrapper`` before training.  Subclasses that extend
        ``StepwiseEstimator`` should resolve their config here (merging
        ``self._init_flat_kwargs`` with the YAML-sourced *config* dict),
        set ``self.loop_config`` and ``self.cache_on_device``, then call
        ``super().setup(simulator_instance, theta_key, condition_keys)``.

        Args:
            simulator_instance: Live prior/simulator, already constructed.
            theta_key: Name of the parameter node being estimated.
            condition_keys: List of evidence/scaffold node names.
            config: YAML-sourced estimator config dict (may be empty ``{}``).
                Overrides kwargs stored from ``__init__``.
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
