"""Gaussian posterior estimation (was SNPE_gaussian).

This module provides:
- GaussianPosterior: nn.Module implementing the Gaussian posterior
- Gaussian: Factory function creating a configured LossBasedEstimator
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from falcon.estimators.networks import build_mlp
from falcon.priors.product import TransformedPrior
from falcon.estimators.stepwise_base import (
    LossBasedEstimator,
    TrainingLoopConfig,
)
from falcon.core.logger import log, debug, info, warning, error


# ==================== Configuration Dataclasses ====================


@dataclass
class NetworkConfig:
    """Configuration for GaussianPosterior network."""

    hidden_dim: int = 128
    num_layers: int = 3
    momentum: float = 0.01
    min_var: float = 1e-20
    eig_update_freq: int = 1


@dataclass
class OptimizerConfig:
    """Optimizer and scheduler parameters."""

    lr: float = 1e-2
    betas: tuple = (0.9, 0.9)  # Lower beta2 for dynamic SBI setting
    lr_decay_factor: float = 1.0  # 1.0 = no LR decay (scheduler disabled)
    scheduler_patience: int = 8


@dataclass
class InferenceConfig:
    """Inference and sampling parameters."""

    gamma: float = 0.5
    discard_samples: bool = False
    log_ratio_threshold: float = -20.0


@dataclass
class GaussianConfig:
    """Top-level Gaussian estimator configuration."""

    loop: TrainingLoopConfig = field(default_factory=TrainingLoopConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
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

        # Input statistics (conditions) - diagonal whitening
        self.register_buffer("_input_mean", torch.zeros(condition_dim))
        self.register_buffer("_input_std", torch.ones(condition_dim))

        # Output statistics (theta) - diagonal whitening
        # Always float64 for precision; results are cast to input dtype on output.
        self.register_buffer("_output_mean", torch.zeros(param_dim, dtype=torch.float64))
        self.register_buffer("_output_std", torch.ones(param_dim, dtype=torch.float64))

        # Residual covariance (prediction error) - full covariance
        self.register_buffer("_residual_cov", torch.eye(param_dim, dtype=torch.float64))
        self.register_buffer("_residual_eigvals", torch.ones(param_dim, dtype=torch.float64))
        self.register_buffer("_residual_eigvecs", torch.eye(param_dim, dtype=torch.float64))

    # ==================== Device/Dtype ====================

    def to(self, *args, **kwargs):
        """Move module, preserving parameter-space buffer dtype."""
        param_dtype = self._output_mean.dtype
        result = super().to(*args, **kwargs)
        for name in ('_output_mean', '_output_std', '_residual_cov',
                     '_residual_eigvals', '_residual_eigvecs'):
            setattr(result, name, getattr(result, name).to(param_dtype))
        return result

    # ==================== Posterior Contract ====================

    def loss(self, theta: torch.Tensor, conditions: torch.Tensor) -> torch.Tensor:
        """Compute negative log likelihood loss, updating statistics only during training."""
        if self.training:
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

        return (-0.5 * (self.param_dim * np.log(2 * np.pi) + log_det + mahal)).to(theta.dtype)

    def sample(self, conditions: torch.Tensor, gamma: Optional[float] = None) -> torch.Tensor:
        """Sample from posterior, optionally tempered.

        Internal computation is float64 (output buffer precision).
        Result is cast to conditions' dtype on return.

        Args:
            conditions: Condition tensor of shape (batch, condition_dim)
            gamma: Tempering parameter. None = untempered, <1 = widened.
        """
        out_dtype = conditions.dtype
        mean = self._forward_mean(conditions)
        V = self._residual_eigvecs
        d = self._residual_eigvals

        if gamma is None:
            eps = torch.randn_like(mean)
            return (mean + (V @ (torch.sqrt(d).unsqueeze(1) * eps.T)).T).to(out_dtype)

        # Tempered sampling
        a = gamma / (1 + gamma)
        lambda_like = (1.0 / d - 1.0).clamp(min=0)
        lambda_prop = a * lambda_like + 1.0
        var_prop = 1.0 / lambda_prop

        mean_proj = V.T @ mean.T
        alpha = a / (d * lambda_prop)
        mean_prop = (V @ (alpha.unsqueeze(1) * mean_proj)).T

        eps = torch.randn_like(mean)
        return (mean_prop + (V @ (torch.sqrt(var_prop).unsqueeze(1) * eps.T)).T).to(out_dtype)

    # ==================== Internal Methods ====================

    def _forward_mean(self, conditions: torch.Tensor) -> torch.Tensor:
        """Predict mean using diagonal whitening.

        Conditions are cast to MLP dtype for input whitening. MLP output
        is cast to parameter-space dtype before de-whitening.
        """
        c = conditions.to(self._input_mean.dtype)
        x_norm = (c - self._input_mean.detach()) / self._input_std.detach()
        r = self.net(x_norm)
        r = r.to(self._output_mean.dtype)
        return self._output_mean.detach() + self._output_std.detach() * r

    def _update_stats(self, theta: torch.Tensor, conditions: torch.Tensor) -> None:
        """Update running statistics using EMA.

        Output buffers are float64 for precision. Input buffers stay in MLP
        dtype (float32); conditions are cast accordingly.
        """
        m = self.momentum
        with torch.no_grad():
            c = conditions.to(self._input_mean.dtype)
            self._input_mean = (1 - m) * self._input_mean + m * c.mean(dim=0)
            batch_input_std = self._compute_std(c, self._input_mean)
            self._input_std = (1 - m) * self._input_std + m * batch_input_std

            self._output_mean = (1 - m) * self._output_mean + m * theta.mean(dim=0)
            batch_output_std = self._compute_std(theta, self._output_mean)
            self._output_std = (1 - m) * self._output_std + m * batch_output_std
            self._output_std = self._output_std.clamp(max=1.0)


    def _update_residual_cov(self, theta: torch.Tensor, conditions: torch.Tensor) -> None:
        """Update residual covariance and eigendecomposition."""
        with torch.no_grad():
            mean = self._forward_mean(conditions)
            residuals = theta - mean

            zero_mean = torch.zeros(self.param_dim, device=theta.device, dtype=theta.dtype)
            batch_cov = self._compute_cov(residuals, zero_mean)
            self._residual_cov = (1 - self.momentum) * self._residual_cov + self.momentum * batch_cov

            self._step_counter += 1
            if self._step_counter % self.eig_update_freq == 0:
                self._update_eigendecomp()
                log({"theta_std": self._output_std.mean().item()})
                log({"residual_eigvals_mean": self._residual_eigvals.mean().item()})

    def _compute_std(self, data: torch.Tensor, mean: torch.Tensor) -> torch.Tensor:
        """Compute per-dimension standard deviation."""
        centered = data - mean
        n = data.shape[0]
        var = (centered ** 2).sum(dim=0) / max(n - 1, 1)
        return torch.sqrt(var.clamp(min=self.min_var))

    def _compute_cov(self, data: torch.Tensor, mean: torch.Tensor) -> torch.Tensor:
        """Compute covariance matrix from data."""
        centered = data - mean
        n = data.shape[0]
        cov = (centered.T @ centered) / max(n - 1, 1)
        eye = torch.eye(data.shape[1], device=data.device, dtype=data.dtype)
        return cov + self.min_var * eye

    def _update_eigendecomp(self) -> None:
        """Update eigendecomposition of residual covariance."""
        eigvals, eigvecs = torch.linalg.eigh(self._residual_cov)
        self._residual_eigvals = eigvals.clamp(min=self.min_var)
        self._residual_eigvecs = eigvecs


# ==================== Factory Function ====================


def Gaussian(
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
            - network: NetworkConfig options
            - optimizer: OptimizerConfig options
            - inference: InferenceConfig options
            - embedding: Embedding configuration with _target_
            - device: Device string (optional)

    Returns:
        Configured LossBasedEstimator ready for training

    Example YAML:
        estimator:
          _target_: falcon.estimators.Gaussian
          network:
            hidden_dim: 128
            num_layers: 3
          embedding:
            _target_: model.E
            _input_: [x]
    """
    # Check simulator supports transformation interface
    if not isinstance(simulator_instance, TransformedPrior):
        raise TypeError(
            f"Gaussian requires a TransformedPrior (e.g., Product), "
            f"got {type(simulator_instance).__name__}. "
            f"The simulator must support forward/inverse with mode='standard_normal'."
        )

    # Merge with defaults
    schema = OmegaConf.structured(GaussianConfig)
    cfg = OmegaConf.merge(schema, config or {})

    # Extract configs as plain dicts
    embedding_config = OmegaConf.to_container(cfg.embedding, resolve=True)
    posterior_config = OmegaConf.to_container(cfg.network, resolve=True)

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
        latent_mode="standard_normal",  # GaussianPosterior assumes N(0,I) prior
    )
