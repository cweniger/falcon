"""Sequential Neural Posterior Estimation with Gaussian posterior (SNPE-Gaussian).

This module provides:
- GaussianPosterior: nn.Module implementing the Gaussian posterior
- SNPE_gaussian: Factory function creating a configured LossBasedEstimator

The Gaussian posterior uses:
- Cholesky-based whitening for input/output normalization
- Eigendecomposition for efficient covariance operations and tempered sampling
- EMA updates for running statistics

Benefits over flow-based approaches:
- Simpler: Full covariance Gaussian is mathematically tractable
- Efficient: Eigendecomposition enables fast sampling and log_prob
- Interpretable: Covariance matrix directly shows parameter correlations
- Tempered proposals: Eigenvalue-based tempering for exploration
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from falcon.contrib.networks import build_mlp
from falcon.contrib.stepwise_estimator import (
    LossBasedEstimator,
    TrainingLoopConfig,
    OptimizerConfig,
    InferenceConfig,
)


# ==================== Configuration Dataclasses ====================


@dataclass
class GaussianPosteriorConfig:
    """Configuration for GaussianPosterior network."""

    hidden_dim: int = 128
    num_layers: int = 3
    momentum: float = 0.01
    min_var: float = 1e-6
    eig_update_freq: int = 1


def _default_optimizer_config():
    return OptimizerConfig(lr=1e-2)


@dataclass
class GaussianConfig:
    """Top-level SNPE_gaussian configuration."""

    loop: TrainingLoopConfig = field(default_factory=TrainingLoopConfig)
    network: GaussianPosteriorConfig = field(default_factory=GaussianPosteriorConfig)
    optimizer: OptimizerConfig = field(default_factory=_default_optimizer_config)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    embedding: Optional[Any] = None
    device: Optional[str] = None


# ==================== GaussianPosterior Module ====================


class GaussianPosterior(nn.Module):
    """Full covariance Gaussian posterior with eigenvalue-based operations.

    Implements the Posterior contract:
        - loss(theta, conditions) -> Tensor
        - sample(conditions, gamma=None) -> Tensor
        - log_prob(theta, conditions) -> Tensor

    The posterior is parameterized as:
        p(theta | conditions) = N(mu(conditions), Sigma)

    where mu(conditions) is predicted by an MLP with whitening, and Sigma is the
    residual covariance matrix estimated from training data.
    """

    def __init__(
        self,
        param_dim: int,
        condition_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        momentum: float = 0.01,
        min_var: float = 1e-6,
        eig_update_freq: int = 1,
    ):
        super().__init__()
        self.param_dim = param_dim
        self.condition_dim = condition_dim
        self.momentum = momentum
        self.min_var = min_var
        self.eig_update_freq = eig_update_freq
        self._step_counter = 0

        # MLP for mean prediction
        self.net = build_mlp(condition_dim, hidden_dim, param_dim, num_layers)

        # Input statistics (conditions)
        self.register_buffer("_input_mean", torch.zeros(condition_dim))
        self.register_buffer("_input_cov", torch.eye(condition_dim))
        self.register_buffer("_input_cov_chol", torch.eye(condition_dim))

        # Output statistics (theta)
        self.register_buffer("_output_mean", torch.zeros(param_dim))
        self.register_buffer("_output_cov", torch.eye(param_dim))
        self.register_buffer("_output_cov_chol", torch.eye(param_dim))

        # Residual covariance (prediction error)
        self.register_buffer("_residual_cov", torch.eye(param_dim))
        self.register_buffer("_residual_eigvals", torch.ones(param_dim))
        self.register_buffer("_residual_eigvecs", torch.eye(param_dim))

    # ==================== Posterior Contract ====================

    def loss(self, theta: torch.Tensor, conditions: torch.Tensor) -> torch.Tensor:
        """Compute negative log likelihood loss, updating statistics."""
        self._update_stats(theta, conditions)
        self._update_residual_cov(theta, conditions)
        log_prob = self.log_prob(theta, conditions)
        return -log_prob.mean()

    def log_prob(self, theta: torch.Tensor, conditions: torch.Tensor) -> torch.Tensor:
        """Compute Gaussian log probability using eigendecomposition."""
        mean = self._forward_mean(conditions)
        residuals = theta - mean

        V = self._residual_eigvecs.detach()
        d = self._residual_eigvals.detach()

        log_det = torch.log(d).sum()
        r_proj = V.T @ residuals.T
        mahal = (r_proj**2 / d.unsqueeze(1)).sum(dim=0)

        return -0.5 * (self.param_dim * np.log(2 * np.pi) + log_det + mahal)

    def sample(self, conditions: torch.Tensor, gamma: Optional[float] = None) -> torch.Tensor:
        """Sample from posterior, optionally tempered.

        Args:
            conditions: Condition tensor of shape (batch, condition_dim)
            gamma: Tempering parameter. None = untempered, <1 = widened.
        """
        mean = self._forward_mean(conditions)
        V = self._residual_eigvecs
        d = self._residual_eigvals

        if gamma is None:
            eps = torch.randn_like(mean)
            return mean + (V @ (torch.sqrt(d).unsqueeze(1) * eps.T)).T

        # Tempered sampling
        a = gamma / (1 + gamma)
        lambda_like = (1.0 / d - 1.0).clamp(min=0)
        lambda_prop = a * lambda_like + 1.0
        var_prop = 1.0 / lambda_prop

        mean_proj = V.T @ mean.T
        alpha = a / (d * lambda_prop)
        mean_prop = (V @ (alpha.unsqueeze(1) * mean_proj)).T

        eps = torch.randn_like(mean)
        return mean_prop + (V @ (torch.sqrt(var_prop).unsqueeze(1) * eps.T)).T

    # ==================== Internal Methods ====================

    def _forward_mean(self, conditions: torch.Tensor) -> torch.Tensor:
        """Predict mean using Cholesky-based whitening."""
        input_mean = self._input_mean.detach()
        input_cov_chol = self._input_cov_chol.detach()
        output_mean = self._output_mean.detach()
        output_cov_chol = self._output_cov_chol.detach()

        centered = (conditions - input_mean).T
        x_white = torch.linalg.solve_triangular(input_cov_chol, centered, upper=False).T
        r = self.net(x_white)
        return output_mean + (output_cov_chol @ r.T).T

    def _update_stats(self, theta: torch.Tensor, conditions: torch.Tensor) -> None:
        """Update running statistics using EMA."""
        with torch.no_grad():
            self._input_mean.lerp_(conditions.mean(dim=0), self.momentum)
            self._output_mean.lerp_(theta.mean(dim=0), self.momentum)

            batch_input_cov = self._compute_cov(conditions, self._input_mean)
            batch_output_cov = self._compute_cov(theta, self._output_mean)
            self._input_cov.lerp_(batch_input_cov, self.momentum)
            self._output_cov.lerp_(batch_output_cov, self.momentum)

            self._input_cov_chol.copy_(self._safe_cholesky(self._input_cov))
            self._output_cov_chol.copy_(self._safe_cholesky(self._output_cov))

    def _update_residual_cov(self, theta: torch.Tensor, conditions: torch.Tensor) -> None:
        """Update residual covariance and eigendecomposition."""
        with torch.no_grad():
            mean = self._forward_mean(conditions)
            residuals = theta - mean

            zero_mean = torch.zeros(self.param_dim, device=theta.device, dtype=theta.dtype)
            batch_cov = self._compute_cov(residuals, zero_mean)
            self._residual_cov.lerp_(batch_cov, self.momentum)

            self._step_counter += 1
            if self._step_counter % self.eig_update_freq == 0:
                self._update_eigendecomp()

    def _compute_cov(self, data: torch.Tensor, mean: torch.Tensor) -> torch.Tensor:
        """Compute covariance matrix from data."""
        centered = data - mean
        n = data.shape[0]
        cov = (centered.T @ centered) / max(n - 1, 1)
        eye = torch.eye(data.shape[1], device=data.device, dtype=data.dtype)
        return cov + self.min_var * eye

    def _safe_cholesky(self, cov: torch.Tensor) -> torch.Tensor:
        """Compute Cholesky with fallback for numerical stability."""
        try:
            return torch.linalg.cholesky(cov)
        except RuntimeError:
            eye = torch.eye(cov.shape[0], device=cov.device, dtype=cov.dtype)
            return torch.linalg.cholesky(cov + 1e-4 * eye)

    def _update_eigendecomp(self) -> None:
        """Update eigendecomposition of residual covariance."""
        eigvals, eigvecs = torch.linalg.eigh(self._residual_cov)
        eigvals = eigvals.clamp(min=self.min_var)
        self._residual_eigvals.copy_(eigvals)
        self._residual_eigvecs.copy_(eigvecs)


# ==================== Factory Function ====================


def SNPE_gaussian(
    simulator_instance,
    theta_key: Optional[str] = None,
    condition_keys: Optional[List[str]] = None,
    config: Optional[dict] = None,
) -> LossBasedEstimator:
    """Create a LossBasedEstimator with GaussianPosterior.

    This is the main entry point for using Gaussian posterior estimation.
    It provides sensible defaults while allowing full customization.

    Args:
        simulator_instance: Prior/simulator instance
        theta_key: Key for theta in batch data
        condition_keys: Keys for condition data in batch
        config: Configuration dict with sections:
            - loop: TrainingLoopConfig options
            - network: GaussianPosteriorConfig options
            - optimizer: OptimizerConfig options
            - inference: InferenceConfig options
            - embedding: Embedding configuration with _target_
            - device: Device string (optional)

    Returns:
        Configured LossBasedEstimator ready for training

    Example YAML:
        estimator:
          _target_: falcon.contrib.SNPE_gaussian
          network:
            hidden_dim: 128
            num_layers: 3
          embedding:
            _target_: model.E
            _input_: [x]
    """
    # Merge with defaults
    schema = OmegaConf.structured(GaussianConfig)
    cfg = OmegaConf.merge(schema, config or {})

    # Extract embedding config
    embedding_config = OmegaConf.to_container(cfg.embedding, resolve=True)

    # Extract posterior config
    posterior_config = {
        "hidden_dim": cfg.network.hidden_dim,
        "num_layers": cfg.network.num_layers,
        "momentum": cfg.network.momentum,
        "min_var": cfg.network.min_var,
        "eig_update_freq": cfg.network.eig_update_freq,
    }

    return LossBasedEstimator(
        simulator_instance=simulator_instance,
        posterior_cls=GaussianPosterior,
        embedding_config=embedding_config,
        loop_config=cfg.loop,
        optimizer_config=cfg.optimizer,
        inference_config=cfg.inference,
        posterior_config=posterior_config,
        theta_key=theta_key,
        condition_keys=condition_keys,
        device=cfg.device,
    )
