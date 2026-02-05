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

    All methods are abstract - no implementation details.
    Concrete implementations must provide all functionality.

    Conditions are passed as Dict[str, Tensor] mapping node names to values.
    Sampling methods return dicts with 'value' (ndarray) and optionally 'log_prob' (ndarray).
    """

    @abstractmethod
    async def train(self, buffer) -> None:
        """
        Train the estimator.

        Args:
            buffer: BufferView providing access to training/validation data
        """
        pass

    @abstractmethod
    def sample_prior(
        self, num_samples: int, conditions: Optional[Conditions] = None
    ) -> dict:
        """
        Sample from the prior distribution.

        Args:
            num_samples: Number of samples to generate
            conditions: Conditioning values from parent nodes (usually None for prior)

        Returns:
            Dict with 'value' (ndarray) and optionally 'log_prob' (ndarray)
        """
        pass

    @abstractmethod
    def sample_posterior(
        self, num_samples: int, conditions: Optional[Conditions] = None
    ) -> dict:
        """
        Sample from the posterior distribution.

        Args:
            num_samples: Number of samples to generate
            conditions: Dict mapping node names to condition tensors

        Returns:
            Dict with 'value' (ndarray) and optionally 'log_prob' (ndarray)
        """
        pass

    @abstractmethod
    def sample_proposal(
        self, num_samples: int, conditions: Optional[Conditions] = None
    ) -> dict:
        """
        Sample from the proposal distribution for adaptive resampling.

        Args:
            num_samples: Number of samples to generate
            conditions: Dict mapping node names to condition tensors

        Returns:
            Dict with 'value' (ndarray) and optionally 'log_prob' (ndarray)
        """
        pass

    @abstractmethod
    def save(self, node_dir: Path) -> None:
        """
        Save estimator state to directory.

        Args:
            node_dir: Directory to save state to
        """
        pass

    @abstractmethod
    def load(self, node_dir: Path) -> None:
        """
        Load estimator state from directory.

        Args:
            node_dir: Directory to load state from
        """
        pass

    @abstractmethod
    def pause(self) -> None:
        """Pause training loop."""
        pass

    @abstractmethod
    def resume(self) -> None:
        """Resume training loop."""
        pass

    @abstractmethod
    def interrupt(self) -> None:
        """Terminate training loop."""
        pass
