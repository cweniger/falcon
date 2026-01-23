"""EmbeddedPosterior wrapper for combining embeddings with posterior networks."""

from typing import Dict

import torch
import torch.nn as nn


class EmbeddedPosterior(nn.Module):
    """Wraps a posterior with an embedding network.

    This composition pattern bundles the embedding and posterior together,
    providing a unified interface that accepts raw condition dicts and handles
    the embedding internally.

    The posterior must implement:
        - loss(theta, conditions): Training loss computation
        - log_prob(theta, conditions): Log probability computation
        - sample(conditions, gamma): Tempered sampling
        - sample_posterior(conditions): Posterior sampling

    Public API mirrors the posterior but accepts Dict[str, Tensor] conditions.
    """

    def __init__(self, embedding: nn.Module, posterior: nn.Module):
        """Initialize EmbeddedPosterior.

        Args:
            embedding: Embedding network that takes Dict[str, Tensor] and returns Tensor
            posterior: Posterior network with loss/log_prob/sample methods
        """
        super().__init__()
        self.embedding = embedding
        self.posterior = posterior

    def _embed(self, conditions: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Run conditions through embedding network."""
        return self.embedding(conditions)

    def loss(self, theta: torch.Tensor, conditions: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute loss with embedded conditions."""
        s = self._embed(conditions)
        return self.posterior.loss(theta, s)

    def log_prob(self, theta: torch.Tensor, conditions: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute log probability with embedded conditions."""
        s = self._embed(conditions)
        return self.posterior.log_prob(theta, s)

    def sample(self, conditions: Dict[str, torch.Tensor], gamma: float = 1.0) -> torch.Tensor:
        """Sample from tempered posterior with embedded conditions."""
        s = self._embed(conditions)
        return self.posterior.sample(s, gamma)

    def sample_posterior(self, conditions: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Sample from posterior with embedded conditions."""
        s = self._embed(conditions)
        return self.posterior.sample_posterior(s)
