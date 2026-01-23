"""Sequential Neural Posterior Estimation with Gaussian posterior (SNPE-Gaussian).

This estimator uses a full covariance Gaussian posterior instead of normalizing flows.
The architecture is based on patterns from gaussian_full_cov.py:
- register_buffer() for explicit state management
- lerp_() for in-place EMA updates of running statistics
- Eigendecomposition for efficient covariance operations and tempered sampling

Benefits over flow-based approaches:
- Simpler: Full covariance Gaussian is mathematically tractable
- Efficient: Eigendecomposition enables fast sampling and log_prob
- Interpretable: Covariance matrix directly shows parameter correlations
- Tempered proposals: Eigenvalue-based tempering for exploration
"""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from falcon.core.utils import RVBatch
from falcon.core.logger import log, debug
from falcon.contrib.stepwise_estimator import (
    LossBasedEstimator,
    TrainingLoopConfig,
    OptimizerConfig,
)
from falcon.contrib.networks import build_mlp
from falcon.contrib.embedded_posterior import EmbeddedPosterior
from falcon.contrib.torch_embedding import instantiate_embedding


# ==================== Configuration Dataclasses ====================


@dataclass
class GaussianNetworkConfig:
    """Neural network architecture parameters for Gaussian posterior."""

    hidden_dim: int = 128
    num_layers: int = 3
    momentum: float = 0.01
    min_var: float = 1e-6
    eig_update_freq: int = 1
    embedding: Optional[Any] = None


@dataclass
class GaussianInferenceConfig:
    """Inference and sampling parameters."""

    gamma: float = 0.5
    discard_samples: bool = True
    log_ratio_threshold: float = -20.0


def _default_optimizer_config():
    """Default optimizer config for SNPE_gaussian (lr=1e-2)."""
    return OptimizerConfig(lr=1e-2)


@dataclass
class GaussianConfig:
    """Top-level SNPE_gaussian configuration."""

    loop: TrainingLoopConfig = field(default_factory=TrainingLoopConfig)
    network: GaussianNetworkConfig = field(default_factory=GaussianNetworkConfig)
    optimizer: OptimizerConfig = field(default_factory=_default_optimizer_config)
    inference: GaussianInferenceConfig = field(default_factory=GaussianInferenceConfig)
    device: Optional[str] = None


# ==================== GaussianPosterior Module ====================


class GaussianPosterior(nn.Module):
    """Full covariance Gaussian posterior with eigenvalue-based operations.

    This module predicts a Gaussian distribution over parameters given conditions.
    It uses:
    - Cholesky-based whitening for input/output normalization
    - Eigendecomposition of residual covariance for efficient log_prob and sampling
    - EMA updates for running statistics

    The posterior is parameterized as:
        p(theta | conditions) = N(mu(conditions), Sigma)

    where mu(conditions) is predicted by an MLP with whitening, and Sigma is the
    residual covariance matrix estimated from training data.

    Public API:
        - loss(theta, conditions): Training entry point, updates stats and returns loss
        - log_prob(theta, conditions): Compute log probability (inference, no side effects)
        - sample(conditions, gamma): Sample from tempered posterior
        - sample_posterior(conditions): Sample from posterior
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

    # ==================== Public API ====================

    def loss(self, theta: torch.Tensor, conditions: torch.Tensor) -> torch.Tensor:
        """Training entry point: update statistics and return negative log likelihood.

        Stats are updated BEFORE computing log_prob to avoid in-place modification
        errors during backward pass. The buffers used in log_prob are detached.

        Args:
            theta: Parameter samples of shape (batch, param_dim)
            conditions: Embedded conditions of shape (batch, condition_dim)

        Returns:
            Scalar loss (negative mean log probability)
        """
        # Update all statistics BEFORE forward pass
        # This avoids in-place modification errors since buffers are detached in log_prob
        self._update_stats(theta, conditions)
        self._update_residual_cov(theta, conditions)

        # Compute log probability (uses detached buffers)
        log_prob = self.log_prob(theta, conditions)
        return -log_prob.mean()

    def log_prob(self, theta: torch.Tensor, conditions: torch.Tensor) -> torch.Tensor:
        """Compute Gaussian log probability using eigendecomposition.

        Pure inference method with no side effects.

        Args:
            theta: Parameter samples of shape (batch, param_dim)
            conditions: Embedded conditions of shape (batch, condition_dim)

        Returns:
            Per-sample log probabilities of shape (batch,)
        """
        mean = self._forward_mean(conditions)
        residuals = theta - mean

        # Detach eigendecomposition buffers to avoid in-place modification errors
        V = self._residual_eigvecs.detach()
        d = self._residual_eigvals.detach()

        # log|Σ| = sum(log(d_i))
        log_det = torch.log(d).sum()

        # Mahalanobis via eigenbasis: sum_i (r_i^2 / d_i) where r_i = (V^T @ residuals)_i
        r_proj = V.T @ residuals.T  # (param_dim, batch)
        mahal = (r_proj**2 / d.unsqueeze(1)).sum(dim=0)  # (batch,)

        return -0.5 * (self.param_dim * np.log(2 * np.pi) + log_det + mahal)

    def sample(self, conditions: torch.Tensor, gamma: Optional[float] = None) -> torch.Tensor:
        """Sample from the posterior, optionally tempered.

        Tempering interpolates between prior (gamma=0) and posterior (gamma=None/inf):
        - gamma=None: Sample from learned posterior N(mu(x), Sigma)
        - gamma in (0,1): Widened proposal for exploration (used in adaptive resampling)
        - gamma=0: Equivalent to prior

        The tempered distribution has precision: gamma * Lambda_like + I
        where Lambda_like = max(Sigma^{-1} - I, 0) is the likelihood precision.

        Args:
            conditions: Condition tensor of shape (batch, condition_dim)
            gamma: Tempering parameter. None means untempered posterior.
                   Lower values give wider distributions.

        Returns:
            Samples of shape (batch, param_dim)
        """
        mean = self._forward_mean(conditions)
        V = self._residual_eigvecs
        d = self._residual_eigvals

        # Untempered posterior: sample from N(mean, residual_cov)
        if gamma is None:
            eps = torch.randn_like(mean)
            return mean + (V @ (torch.sqrt(d).unsqueeze(1) * eps.T)).T

        # Tempered sampling
        a = gamma / (1 + gamma)

        # Likelihood precision eigenvalues: max(1/d - 1, 0)
        lambda_like = (1.0 / d - 1.0).clamp(min=0)

        # Proposal precision and variance eigenvalues
        lambda_prop = a * lambda_like + 1.0
        var_prop = 1.0 / lambda_prop

        # Mean shrinkage in eigenbasis
        mean_proj = V.T @ mean.T  # (param_dim, batch)
        alpha = a / (d * lambda_prop)
        mean_prop = (V @ (alpha.unsqueeze(1) * mean_proj)).T  # (batch, param_dim)

        # Sample
        eps = torch.randn_like(mean)
        return mean_prop + (V @ (torch.sqrt(var_prop).unsqueeze(1) * eps.T)).T

    # ==================== Internal Methods ====================

    def _forward_mean(self, conditions: torch.Tensor) -> torch.Tensor:
        """Predict mean using Cholesky-based whitening."""
        # Detach buffers to avoid in-place modification errors during backward
        input_mean = self._input_mean.detach()
        input_cov_chol = self._input_cov_chol.detach()
        output_mean = self._output_mean.detach()
        output_cov_chol = self._output_cov_chol.detach()

        # Whiten inputs
        centered = (conditions - input_mean).T
        x_white = torch.linalg.solve_triangular(
            input_cov_chol, centered, upper=False
        ).T

        # Apply MLP
        r = self.net(x_white)

        # De-whiten outputs
        return output_mean + (output_cov_chol @ r.T).T

    def _update_stats(self, theta: torch.Tensor, conditions: torch.Tensor) -> None:
        """Update running statistics using lerp_() for EMA updates."""
        with torch.no_grad():
            # Update means
            self._input_mean.lerp_(conditions.mean(dim=0), self.momentum)
            self._output_mean.lerp_(theta.mean(dim=0), self.momentum)

            # Update covariances
            batch_input_cov = self._compute_cov(conditions, self._input_mean)
            batch_output_cov = self._compute_cov(theta, self._output_mean)
            self._input_cov.lerp_(batch_input_cov, self.momentum)
            self._output_cov.lerp_(batch_output_cov, self.momentum)

            # Update Cholesky factors
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

            # Update eigendecomposition at specified frequency
            self._step_counter += 1
            if self._step_counter % self.eig_update_freq == 0:
                self._update_eigendecomp()

    def _compute_cov(self, data: torch.Tensor, mean: torch.Tensor) -> torch.Tensor:
        """Compute covariance matrix from data."""
        centered = data - mean
        n = data.shape[0]
        cov = (centered.T @ centered) / max(n - 1, 1)
        # Add minimum variance regularization
        eye = torch.eye(data.shape[1], device=data.device, dtype=data.dtype)
        cov = cov + self.min_var * eye
        return cov

    def _safe_cholesky(self, cov: torch.Tensor) -> torch.Tensor:
        """Compute Cholesky decomposition with fallback for numerical stability."""
        try:
            return torch.linalg.cholesky(cov)
        except RuntimeError:
            # Add small regularization and retry
            eye = torch.eye(cov.shape[0], device=cov.device, dtype=cov.dtype)
            cov = cov + 1e-4 * eye
            return torch.linalg.cholesky(cov)

    def _update_eigendecomp(self) -> None:
        """Update eigendecomposition of residual covariance."""
        eigvals, eigvecs = torch.linalg.eigh(self._residual_cov)
        # Clamp eigenvalues for numerical stability
        eigvals = eigvals.clamp(min=self.min_var)
        self._residual_eigvals.copy_(eigvals)
        self._residual_eigvecs.copy_(eigvecs)


# ==================== SNPE_gaussian Implementation ====================


class SNPE_gaussian(LossBasedEstimator):
    """Sequential Neural Posterior Estimation with Gaussian posterior.

    This estimator learns a full covariance Gaussian posterior q(theta|x)
    instead of using normalizing flows. The advantages are:

    1. Simplicity: Gaussian is analytically tractable
    2. Interpretability: Covariance matrix shows parameter correlations
    3. Efficient sampling: Eigendecomposition enables fast sampling
    4. Tempered proposals: Eigenvalue-based tempering for exploration
    """

    def __init__(
        self,
        simulator_instance,
        theta_key: Optional[str] = None,
        condition_keys: Optional[List[str]] = None,
        config: Optional[dict] = None,
    ):
        """Initialize SNPE_gaussian estimator.

        Args:
            simulator_instance: Prior/simulator instance
            theta_key: Key for theta in batch data
            condition_keys: Keys for condition data in batch
            config: Configuration dict with loop, network, optimizer, inference sections
        """
        # Merge user config with defaults using OmegaConf structured config
        schema = OmegaConf.structured(GaussianConfig)
        config = OmegaConf.merge(schema, config or {})

        super().__init__(
            simulator_instance=simulator_instance,
            loop_config=config.loop,
            optimizer_config=config.optimizer,
            theta_key=theta_key,
            condition_keys=condition_keys,
            device=config.device,
        )

        self.config = config

        # Stored tensors from first batch (needed for model rebuild on load)
        self._init_theta: Optional[torch.Tensor] = None
        self._init_conditions: Optional[Dict[str, torch.Tensor]] = None

        # Extended history for tracking
        self.history.update({
            "theta_mins": [],
            "theta_maxs": [],
        })

    # ==================== LossBasedEstimator Abstract Methods ====================

    def _build_model(self, batch) -> nn.Module:
        """Build EmbeddedPosterior model from first batch."""
        # Extract and store tensors for reload
        self._init_theta = torch.from_numpy(batch[self.theta_key])
        self._init_conditions = {k: torch.from_numpy(batch[k]) for k in self.condition_keys if k in batch}
        return self._create_model(self._init_theta, self._init_conditions)

    def _create_model(self, theta: torch.Tensor, conditions: Dict[str, torch.Tensor]) -> nn.Module:
        """Create EmbeddedPosterior from theta and conditions tensors."""
        debug("Building model...")
        cfg_net = self.config.network

        # Create embedding and infer condition_dim
        embedding_config = OmegaConf.to_container(cfg_net.embedding, resolve=True)
        embedding = instantiate_embedding(embedding_config).to(self.device)
        embedding.eval()
        with torch.no_grad():
            conditions_device = {k: v.to(self.device) for k, v in conditions.items()}
            embedded = embedding(conditions_device)

        # Create posterior
        posterior = GaussianPosterior(
            param_dim=theta.shape[1],
            condition_dim=embedded.shape[1],
            hidden_dim=cfg_net.hidden_dim,
            num_layers=cfg_net.num_layers,
            momentum=cfg_net.momentum,
            min_var=cfg_net.min_var,
            eig_update_freq=cfg_net.eig_update_freq,
        ).to(self.device)

        debug("Model built.")
        return EmbeddedPosterior(embedding, posterior)

    def _compute_loss(self, batch) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute loss from batch."""
        # Extract and convert to tensors
        theta = torch.from_numpy(batch[self.theta_key]).to(self.device, dtype=torch.float32)
        theta_logprob = torch.from_numpy(batch[f"{self.theta_key}.logprob"])
        conditions = {
            k: torch.from_numpy(batch[k]).to(self.device, dtype=torch.float32)
            for k in self.condition_keys if k in batch
        }

        # Record sample IDs for history
        ts = time.time()
        self.history["train_ids"].extend((ts, id) for id in batch._ids.tolist())

        # Track theta ranges
        with torch.no_grad():
            self.history["theta_mins"].append(theta.min(dim=0).values.cpu().numpy())
            self.history["theta_maxs"].append(theta.max(dim=0).values.cpu().numpy())

        # Compute loss (stats updated inside loss() before log_prob)
        loss = self._model.loss(theta, conditions)

        # Discard low-probability samples
        if self.config.inference.discard_samples:
            with torch.no_grad():
                self._model.eval()
                log_prob = self._model.log_prob(theta, conditions).cpu()
            log_ratio = log_prob - theta_logprob
            discard_mask = log_ratio < self.config.inference.log_ratio_threshold
            batch.discard(discard_mask)

        return loss, {"loss": loss.item()}

    # ==================== Sampling Methods ====================

    def sample_prior(self, num_samples: int, conditions: Optional[Dict] = None) -> RVBatch:
        """Sample from the prior distribution."""
        if conditions:
            raise ValueError("Conditions are not supported for sample_prior.")
        samples = self.simulator_instance.simulate_batch(num_samples)
        logprob = np.zeros(num_samples)
        return RVBatch(samples, logprob=logprob)

    def _sample(self, num_samples: int, conditions: Optional[Dict], gamma: Optional[float]) -> RVBatch:
        """Internal sampling method using inference model.

        Falls back to sample_prior if model not yet available.

        Args:
            num_samples: Number of samples to generate
            conditions: Dict mapping node names to condition tensors
            gamma: Tempering parameter (None for untempered posterior)
        """
        if not self.has_trained_model:
            return self.sample_prior(num_samples)

        assert conditions, "Conditions must be provided for sampling."

        # Move conditions to device and expand for num_samples
        conditions_device = {
            k: v.to(self.device, dtype=torch.float32).expand(num_samples, *v.shape[1:])
            for k, v in conditions.items()
        }

        model = self.inference_model
        with torch.no_grad():
            model.eval()
            samples = model.sample(conditions_device, gamma=gamma)
            logprob = model.log_prob(samples, conditions_device)

        return RVBatch(samples.cpu().numpy(), logprob=logprob.cpu().numpy())

    def sample_posterior(self, num_samples: int, conditions: Optional[Dict] = None) -> RVBatch:
        """Sample from the posterior distribution q(theta|x)."""
        return self._sample(num_samples, conditions, gamma=None)

    def sample_proposal(self, num_samples: int, conditions: Optional[Dict] = None) -> RVBatch:
        """Sample from widened proposal distribution for adaptive resampling."""
        result = self._sample(num_samples, conditions, gamma=self.config.inference.gamma)
        log({
            "sample_proposal:mean": result.value.mean(),
            "sample_proposal:std": result.value.std(),
            "sample_proposal:logprob": result.logprob.mean(),
        })
        return result

    # ==================== Save/Load ====================

    def save(self, node_dir: Path) -> None:
        """Save SNPE_gaussian state."""
        debug(f"Saving: {node_dir}")
        super().save(node_dir)

        # Save init tensors for model rebuild
        torch.save({"theta": self._init_theta, "conditions": self._init_conditions},
                   node_dir / "init_tensors.pth")
        torch.save(self.history["theta_mins"], node_dir / "theta_mins_batches.pth")
        torch.save(self.history["theta_maxs"], node_dir / "theta_maxs_batches.pth")

    def _rebuild_model_for_load(self, node_dir: Path) -> nn.Module:
        """Rebuild model from saved init tensors."""
        data = torch.load(node_dir / "init_tensors.pth")
        self._init_theta = data["theta"]
        self._init_conditions = data["conditions"]
        return self._create_model(self._init_theta, self._init_conditions)

    def load(self, node_dir: Path) -> None:
        """Load SNPE_gaussian state."""
        debug(f"Loading: {node_dir}")
        super().load(node_dir)
