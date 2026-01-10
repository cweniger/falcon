"""Abstract base class for inference estimators."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

from falcon.core.utils import RVBatch


class BaseEstimator(ABC):
    """
    Fully abstract base class defining the estimator interface.

    All methods are abstract - no implementation details.
    Concrete implementations must provide all functionality.
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
        self, num_samples: int, parent_conditions: List = []
    ) -> RVBatch:
        """
        Sample from the prior distribution.

        Args:
            num_samples: Number of samples to generate
            parent_conditions: Conditioning values from parent nodes

        Returns:
            RVBatch with samples and log probabilities
        """
        pass

    @abstractmethod
    def sample_posterior(
        self,
        num_samples: int,
        parent_conditions: List = [],
        evidence_conditions: List = [],
    ) -> RVBatch:
        """
        Sample from the posterior distribution.

        Args:
            num_samples: Number of samples to generate
            parent_conditions: Conditioning values from parent nodes
            evidence_conditions: Observed evidence values

        Returns:
            RVBatch with samples and log probabilities
        """
        pass

    @abstractmethod
    def sample_proposal(
        self,
        num_samples: int,
        parent_conditions: List = [],
        evidence_conditions: List = [],
    ) -> RVBatch:
        """
        Sample from the proposal distribution for adaptive resampling.

        Args:
            num_samples: Number of samples to generate
            parent_conditions: Conditioning values from parent nodes
            evidence_conditions: Observed evidence values

        Returns:
            RVBatch with samples and log probabilities
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
