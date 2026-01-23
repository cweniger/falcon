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
from falcon.core.logger import log, debug, info, warning, error
from falcon.contrib.stepwise_estimator import (
    LossBasedEstimator,
    TrainingLoopConfig,
    OptimizerConfig,
)
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
    use_best_model_during_inference: bool = True


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


# ==================== Helper Functions ====================


def build_mlp(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    num_layers: int,
) -> nn.Sequential:
    """Build MLP with specified architecture."""
    layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
    for _ in range(num_layers - 1):
        layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
    layers.append(nn.Linear(hidden_dim, output_dim))
    return nn.Sequential(*layers)


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
        """Compute negative log likelihood loss.

        NOTE: This method only computes the loss. Call update_stats() separately
        after loss.backward() to update running statistics.

        Args:
            theta: Parameter samples of shape (batch, param_dim)
            conditions: Embedded conditions of shape (batch, condition_dim)

        Returns:
            Scalar loss (negative mean log probability)
        """
        # Compute log probability (no stats updates here to avoid in-place issues)
        log_prob = self.log_prob(theta, conditions)
        return -log_prob.mean()

    def update_stats(self, theta: torch.Tensor, conditions: torch.Tensor) -> None:
        """Update running statistics after backward pass.

        Call this AFTER loss.backward() to avoid in-place modification errors.

        Args:
            theta: Parameter samples of shape (batch, param_dim)
            conditions: Embedded conditions of shape (batch, condition_dim)
        """
        self._update_stats(theta, conditions)
        self._update_residual_cov(theta, conditions)

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

        # Detach eigendecomposition to avoid in-place modification errors
        # (these buffers are updated after loss computation in training)
        V = self._residual_eigvecs.detach()
        d = self._residual_eigvals.detach()

        # log|Σ| = sum(log(d_i))
        log_det = torch.log(d).sum()

        # Mahalanobis via eigenbasis: sum_i (r_i^2 / d_i) where r_i = (V^T @ residuals)_i
        r_proj = V.T @ residuals.T  # (param_dim, batch)
        mahal = (r_proj**2 / d.unsqueeze(1)).sum(dim=0)  # (batch,)

        return -0.5 * (self.param_dim * np.log(2 * np.pi) + log_det + mahal)

    def sample(self, conditions: torch.Tensor, gamma: float = 1.0) -> torch.Tensor:
        """Sample from tempered posterior.

        The proposal precision is: gamma * lambda_like + 1
        where lambda_like = max(1/d - 1, 0) and d are posterior covariance eigenvalues.

        When gamma=1, this samples from the posterior.
        When gamma<1, this samples from a wider proposal distribution.

        Args:
            conditions: Condition tensor of shape (batch, condition_dim)
            gamma: Tempering parameter. gamma=1 gives posterior, gamma<1 gives wider proposal.

        Returns:
            Samples of shape (batch, param_dim)
        """
        mean = self._forward_mean(conditions)
        d = self._residual_eigvals  # posterior covariance eigenvalues
        V = self._residual_eigvecs

        a = gamma / (1 + gamma)

        # Likelihood precision eigenvalues: max(1/d - 1, 0)
        # Truncate at 0 for directions where posterior is wider than prior
        lambda_like = (1.0 / d - 1.0).clamp(min=0)

        # Proposal precision eigenvalues: a * lambda_like + 1
        lambda_prop = a * lambda_like + 1.0

        # Proposal variance eigenvalues
        var_prop = 1.0 / lambda_prop

        # Mean shrinkage in eigenbasis
        # alpha_i = a / (d_i * lambda_prop_i)
        mean_proj = V.T @ mean.T  # (param_dim, batch)
        alpha = a / (d * lambda_prop)
        mean_prop = (V @ (alpha.unsqueeze(1) * mean_proj)).T  # (batch, param_dim)

        # Sample: mean_prop + V @ diag(sqrt(var_prop)) @ eps
        eps = torch.randn_like(mean)
        return mean_prop + (V @ (torch.sqrt(var_prop).unsqueeze(1) * eps.T)).T

    def sample_posterior(self, conditions: torch.Tensor) -> torch.Tensor:
        """Sample from the posterior distribution (gamma -> infinity equivalent).

        Args:
            conditions: Condition tensor of shape (batch, condition_dim)

        Returns:
            Samples of shape (batch, param_dim)
        """
        mean = self._forward_mean(conditions)
        V = self._residual_eigvecs
        d = self._residual_eigvals

        # Sample from N(mean, residual_cov)
        eps = torch.randn_like(mean)
        return mean + (V @ (torch.sqrt(d).unsqueeze(1) * eps.T)).T

    # ==================== Internal Methods ====================

    def _forward_mean(self, conditions: torch.Tensor) -> torch.Tensor:
        """Predict mean using Cholesky-based whitening."""
        # Detach buffers that are updated in-place during training
        # to avoid "modified by inplace operation" errors during backward
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


# ==================== EmbeddedPosterior Wrapper ====================


class EmbeddedPosterior(nn.Module):
    """Wraps a posterior with an embedding network.

    This composition pattern bundles the embedding and posterior together,
    providing a unified interface that accepts raw condition dicts and handles
    the embedding internally.

    Public API mirrors GaussianPosterior but accepts Dict[str, Tensor] conditions:
        - loss(theta, conditions): Training entry point
        - log_prob(theta, conditions): Compute log probability
        - sample(conditions, gamma): Sample from tempered posterior
        - sample_posterior(conditions): Sample from posterior
    """

    def __init__(self, embedding: nn.Module, posterior: GaussianPosterior):
        super().__init__()
        self.embedding = embedding
        self.posterior = posterior

    def _embed(self, conditions: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Run conditions through embedding network."""
        return self.embedding(conditions)

    def loss(self, theta: torch.Tensor, conditions: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Training entry point with embedded conditions."""
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

    def update_stats(self, theta: torch.Tensor, conditions: Dict[str, torch.Tensor]) -> None:
        """Update running statistics with embedded conditions (call after backward)."""
        with torch.no_grad():
            s = self._embed(conditions)
            self.posterior.update_stats(theta, s)


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

        # Embedding network (created now, wrapped with posterior later)
        embedding_config = OmegaConf.to_container(config.network.embedding, resolve=True)
        self._embedding = instantiate_embedding(embedding_config).to(self.device)

        # Store init parameters for save/load
        self._init_parameters = None

        # Extended history for tracking
        self.history.update({
            "theta_mins": [],
            "theta_maxs": [],
        })

    # ==================== LossBasedEstimator Abstract Methods ====================

    def _build_model(self, batch) -> nn.Module:
        """Build EmbeddedPosterior model from first batch."""
        theta, conditions = self._extract_theta_conditions(batch)
        self._init_parameters = [theta, conditions]
        return self._build_model_from_params(theta, conditions)

    def _build_model_from_params(self, theta: torch.Tensor, conditions: Dict) -> nn.Module:
        """Build model from theta and conditions tensors.

        Used both during training (from batch) and loading (from stored params).
        """
        debug("Building model...")
        debug(f"GPU available: {torch.cuda.is_available()}")

        cfg_net = self.config.network

        # Embed conditions to get embedding dimension
        conditions_device = {k: v.to(self.device) for k, v in conditions.items()}
        self._embedding.eval()
        with torch.no_grad():
            s = self._embedding(conditions_device)
        theta_device = theta.to(self.device)

        param_dim = theta_device.shape[1]
        condition_dim = s.shape[1]

        # Create posterior network
        posterior = GaussianPosterior(
            param_dim=param_dim,
            condition_dim=condition_dim,
            hidden_dim=cfg_net.hidden_dim,
            num_layers=cfg_net.num_layers,
            momentum=cfg_net.momentum,
            min_var=cfg_net.min_var,
            eig_update_freq=cfg_net.eig_update_freq,
        ).to(self.device)

        # Wrap embedding + posterior into EmbeddedPosterior
        model = EmbeddedPosterior(self._embedding, posterior)

        debug("Model built.")
        return model

    def _compute_loss(self, batch) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute loss from batch.

        Returns:
            Tuple of (loss_tensor, metrics_dict)
        """
        ids, theta, theta_logprob, conditions, theta_device, conditions_device = \
            self._unpack_batch(batch, "train")

        # Track theta ranges
        with torch.no_grad():
            self.history["theta_mins"].append(theta.min(dim=0).values.cpu().numpy())
            self.history["theta_maxs"].append(theta.max(dim=0).values.cpu().numpy())

        # Compute loss using EmbeddedPosterior.loss()
        loss = self._model.loss(theta_device, conditions_device)

        # Discard samples based on log-likelihood ratio
        if self.config.inference.discard_samples:
            discard_mask = self._compute_discard_mask(theta, theta_logprob, conditions_device)
            batch.discard(discard_mask)

        return loss, {"loss": loss.item()}

    # ==================== Batch Processing ====================

    def _extract_theta_conditions(self, batch) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Extract theta and conditions from batch (without logging)."""
        theta = torch.from_numpy(batch[self.theta_key])
        conditions = {
            k: torch.from_numpy(batch[k]) for k in self.condition_keys if k in batch
        }
        return theta, conditions

    def _unpack_batch(self, batch, phase: str):
        """Unpack batch data and convert to tensors.

        Args:
            batch: Batch object with theta, logprob, and conditions
            phase: "train" or "val" for logging and history

        Returns:
            Tuple of (ids, theta, theta_logprob, conditions, theta_device, conditions_device)
        """
        ids = batch._ids
        theta = torch.from_numpy(batch[self.theta_key])
        theta_logprob = torch.from_numpy(batch[f"{self.theta_key}.logprob"])
        conditions = {
            k: torch.from_numpy(batch[k]) for k in self.condition_keys if k in batch
        }

        # Record IDs for history
        ts = time.time()
        self.history[f"{phase}_ids"].extend((ts, id) for id in ids.tolist())

        log({f"{phase}:theta_logprob_min": theta_logprob.min().item()})
        log({f"{phase}:theta_logprob_max": theta_logprob.max().item()})

        # Move to device and ensure float32 dtype
        conditions_device = {k: v.to(self.device, dtype=torch.float32) for k, v in conditions.items()}
        theta_device = theta.to(self.device, dtype=torch.float32)

        return ids, theta, theta_logprob, conditions, theta_device, conditions_device

    def train_step(self, batch) -> Dict[str, float]:
        """SNPE_gaussian training step with gradient update and stats update.

        Override base class to call update_stats() after backward, avoiding
        in-place modification errors.
        """
        # Initialize on first batch
        if not self.networks_initialized:
            self._initialize_model(batch)

        # Unpack batch
        ids, theta, theta_logprob, conditions, theta_device, conditions_device = \
            self._unpack_batch(batch, "train")

        # Track theta ranges
        with torch.no_grad():
            self.history["theta_mins"].append(theta.min(dim=0).values.cpu().numpy())
            self.history["theta_maxs"].append(theta.max(dim=0).values.cpu().numpy())

        # Forward pass and backward
        self._optimizer.zero_grad()
        self._model.train()
        loss = self._model.loss(theta_device, conditions_device)
        loss.backward()
        self._optimizer.step()

        # Update running statistics AFTER backward to avoid in-place errors
        with torch.no_grad():
            self._model.update_stats(theta_device, conditions_device)

        # Discard samples based on log-likelihood ratio
        if self.config.inference.discard_samples:
            discard_mask = self._compute_discard_mask(theta, theta_logprob, conditions_device)
            batch.discard(discard_mask)

        return {"loss": loss.item()}

    def val_step(self, batch) -> Dict[str, float]:
        """SNPE_gaussian validation step without gradient computation.

        Override base class to use log_prob directly instead of loss().
        """
        _, theta, theta_logprob, conditions, theta_device, conditions_device = \
            self._unpack_batch(batch, "val")

        # Compute loss without gradients
        with torch.no_grad():
            self._model.eval()
            log_prob = self._model.log_prob(theta_device, conditions_device)
            loss = -log_prob.mean()

        return {"loss": loss.item()}

    # ==================== Sampling Methods ====================

    def sample_prior(self, num_samples: int, parent_conditions: Optional[List] = None) -> RVBatch:
        """Sample from the prior distribution."""
        if parent_conditions:
            raise ValueError("Conditions are not supported for sample_prior.")
        samples = self.simulator_instance.simulate_batch(num_samples)
        # Compute log probability from simulator
        logprob = np.zeros(num_samples)  # Prior logprob handled by simulator
        return RVBatch(samples, logprob=logprob)

    def sample_posterior(
        self,
        num_samples: int,
        parent_conditions: Optional[List] = None,
        evidence_conditions: Optional[List] = None,
    ) -> RVBatch:
        """Sample from the posterior distribution q(theta|x)."""
        # Fall back to prior if networks not yet initialized
        if not self.networks_initialized:
            return self.sample_prior(num_samples, parent_conditions)

        cfg_inf = self.config.inference

        conditions_list = (parent_conditions or []) + (evidence_conditions or [])
        assert conditions_list, "Conditions must be provided for posterior sampling."

        # Convert list to dict using condition_keys (ensure float32)
        conditions = {
            k: v.to(self.device, dtype=torch.float32) for k, v in zip(self.condition_keys, conditions_list)
        }

        # Expand conditions for num_samples
        conditions = {k: v.expand(num_samples, *v.shape[1:]) for k, v in conditions.items()}

        # Use best model if configured
        model = self._best_model if cfg_inf.use_best_model_during_inference else self._model

        with torch.no_grad():
            model.eval()
            samples = model.sample_posterior(conditions)
            logprob = model.log_prob(samples, conditions)

        return RVBatch(samples.cpu().numpy(), logprob=logprob.cpu().numpy())

    def sample_proposal(
        self,
        num_samples: int,
        parent_conditions: Optional[List] = None,
        evidence_conditions: Optional[List] = None,
    ) -> RVBatch:
        """Sample from the widened proposal distribution for adaptive resampling.

        Uses eigenvalue-based tempering to widen the posterior for exploration.
        """
        # Fall back to prior if networks not yet initialized
        if not self.networks_initialized:
            return self.sample_prior(num_samples, parent_conditions)

        cfg_inf = self.config.inference

        conditions_list = (parent_conditions or []) + (evidence_conditions or [])
        assert conditions_list, "Conditions must be provided for proposal sampling."

        # Convert list to dict using condition_keys (ensure float32)
        conditions = {
            k: v.to(self.device, dtype=torch.float32) for k, v in zip(self.condition_keys, conditions_list)
        }

        # Expand conditions for num_samples
        conditions = {k: v.expand(num_samples, *v.shape[1:]) for k, v in conditions.items()}

        # Use best model if configured
        model = self._best_model if cfg_inf.use_best_model_during_inference else self._model

        with torch.no_grad():
            model.eval()
            samples = model.sample(conditions, gamma=cfg_inf.gamma)
            logprob = model.log_prob(samples, conditions)

        log({
            "sample_proposal:mean": samples.mean().item(),
            "sample_proposal:std": samples.std().item(),
            "sample_proposal:logprob": logprob.mean().item(),
        })

        return RVBatch(samples.cpu().numpy(), logprob=logprob.cpu().numpy())

    # ==================== Save/Load ====================

    def save(self, node_dir: Path) -> None:
        """Save SNPE_gaussian state."""
        debug(f"Saving: {node_dir}")
        if not self.networks_initialized:
            raise RuntimeError("Networks not initialized.")

        # Save best model state
        torch.save(self._best_model.state_dict(), node_dir / "model.pth")
        torch.save(self._init_parameters, node_dir / "init_parameters.pth")

        # Save history
        torch.save(self.history["train_ids"], node_dir / "train_id_history.pth")
        torch.save(self.history["val_ids"], node_dir / "validation_id_history.pth")
        torch.save(self.history["theta_mins"], node_dir / "theta_mins_batches.pth")
        torch.save(self.history["theta_maxs"], node_dir / "theta_maxs_batches.pth")
        torch.save(self.history["epochs"], node_dir / "epochs.pth")
        torch.save(self.history["train_loss"], node_dir / "loss_train_posterior.pth")
        torch.save(self.history["val_loss"], node_dir / "loss_val_posterior.pth")
        torch.save(self.history["n_samples"], node_dir / "n_samples_total.pth")
        torch.save(self.history["elapsed_min"], node_dir / "elapsed_minutes.pth")

    def load(self, node_dir: Path) -> None:
        """Load SNPE_gaussian state."""
        debug(f"Loading: {node_dir}")
        init_parameters = torch.load(node_dir / "init_parameters.pth")
        theta, conditions = init_parameters[0], init_parameters[1]

        # Build model from stored init parameters
        self._init_parameters = [theta, conditions]
        self._model = self._build_model_from_params(theta, conditions)
        self._best_model = self._clone_model(self._model)

        # Setup optimizer and scheduler (from LossBasedEstimator)
        cfg = self.optimizer_config
        from torch.optim import AdamW
        from torch.optim.lr_scheduler import ReduceLROnPlateau
        self._optimizer = AdamW(self._model.parameters(), lr=cfg.lr)
        self._scheduler = ReduceLROnPlateau(
            self._optimizer,
            mode="min",
            factor=cfg.lr_decay_factor,
            patience=cfg.scheduler_patience,
        )

        self.networks_initialized = True

        # Load best model weights
        self._best_model.load_state_dict(torch.load(node_dir / "model.pth"))

    # ==================== Private Helpers ====================

    def _compute_discard_mask(
        self, theta: torch.Tensor, theta_logprob: torch.Tensor, conditions: Dict
    ):
        """Compute boolean mask of samples to discard based on log-likelihood ratio."""
        cfg_inf = self.config.inference

        theta_device = theta.to(self.device, dtype=torch.float32)

        with torch.no_grad():
            self._model.eval()
            log_prob = self._model.log_prob(theta_device, conditions).cpu()

        log_ratio = log_prob - theta_logprob.cpu()
        return log_ratio < cfg_inf.log_ratio_threshold
