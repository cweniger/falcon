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

import copy
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from falcon.core.utils import RVBatch
from falcon.core.logger import log, debug, info, warning, error
from falcon.contrib.stepwise_estimator import StepwiseEstimator, TrainingLoopConfig
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
class GaussianOptimizerConfig:
    """Optimizer parameters."""

    lr: float = 1e-2
    lr_decay_factor: float = 0.1
    scheduler_patience: int = 8


@dataclass
class GaussianInferenceConfig:
    """Inference and sampling parameters."""

    gamma: float = 0.5
    discard_samples: bool = True
    log_ratio_threshold: float = -20.0
    use_best_model_during_inference: bool = True


@dataclass
class GaussianConfig:
    """Top-level SNPE_gaussian configuration."""

    loop: TrainingLoopConfig = field(default_factory=TrainingLoopConfig)
    network: GaussianNetworkConfig = field(default_factory=GaussianNetworkConfig)
    optimizer: GaussianOptimizerConfig = field(default_factory=GaussianOptimizerConfig)
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
        self.step_counter = 0

        # MLP for mean prediction
        self.net = build_mlp(condition_dim, hidden_dim, param_dim, num_layers)

        # Input statistics (conditions)
        self.register_buffer("input_mean", torch.zeros(condition_dim))
        self.register_buffer("input_cov", torch.eye(condition_dim))
        self.register_buffer("input_cov_chol", torch.eye(condition_dim))

        # Output statistics (theta)
        self.register_buffer("output_mean", torch.zeros(param_dim))
        self.register_buffer("output_cov", torch.eye(param_dim))
        self.register_buffer("output_cov_chol", torch.eye(param_dim))

        # Residual covariance (prediction error)
        self.register_buffer("residual_cov", torch.eye(param_dim))
        self.register_buffer("residual_eigvals", torch.ones(param_dim))
        self.register_buffer("residual_eigvecs", torch.eye(param_dim))

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
        eigvals, eigvecs = torch.linalg.eigh(self.residual_cov)
        # Clamp eigenvalues for numerical stability
        eigvals = eigvals.clamp(min=self.min_var)
        self.residual_eigvals.copy_(eigvals)
        self.residual_eigvecs.copy_(eigvecs)

    def update_stats(self, theta: torch.Tensor, conditions: torch.Tensor) -> None:
        """Update running statistics using lerp_() for EMA updates."""
        with torch.no_grad():
            # Update means
            self.input_mean.lerp_(conditions.mean(dim=0), self.momentum)
            self.output_mean.lerp_(theta.mean(dim=0), self.momentum)

            # Update covariances
            batch_input_cov = self._compute_cov(conditions, self.input_mean)
            batch_output_cov = self._compute_cov(theta, self.output_mean)
            self.input_cov.lerp_(batch_input_cov, self.momentum)
            self.output_cov.lerp_(batch_output_cov, self.momentum)

            # Update Cholesky factors
            self.input_cov_chol.copy_(self._safe_cholesky(self.input_cov))
            self.output_cov_chol.copy_(self._safe_cholesky(self.output_cov))

    def update_residual_cov(self, theta: torch.Tensor, conditions: torch.Tensor) -> None:
        """Update residual covariance and eigendecomposition."""
        with torch.no_grad():
            mean = self.forward_mean(conditions)
            residuals = theta - mean

            zero_mean = torch.zeros(self.param_dim, device=theta.device, dtype=theta.dtype)
            batch_cov = self._compute_cov(residuals, zero_mean)
            self.residual_cov.lerp_(batch_cov, self.momentum)

            # Update eigendecomposition at specified frequency
            self.step_counter += 1
            if self.step_counter % self.eig_update_freq == 0:
                self._update_eigendecomp()

    def forward_mean(self, conditions: torch.Tensor) -> torch.Tensor:
        """Predict mean using Cholesky-based whitening.

        Whitening helps with optimization by normalizing the input/output scales.
        """
        # Whiten inputs
        centered = (conditions - self.input_mean).T
        x_white = torch.linalg.solve_triangular(
            self.input_cov_chol, centered, upper=False
        ).T

        # Apply MLP
        r = self.net(x_white)

        # De-whiten outputs
        return self.output_mean + (self.output_cov_chol @ r.T).T

    def log_prob(self, theta: torch.Tensor, conditions: torch.Tensor) -> torch.Tensor:
        """Compute Gaussian log probability using eigendecomposition.

        Returns per-sample log probabilities.
        log_det is detached since covariance is updated separately via EMA.
        """
        mean = self.forward_mean(conditions)
        residuals = theta - mean

        # log|Σ| = sum(log(d_i))
        log_det = torch.log(self.residual_eigvals).sum()

        # Mahalanobis via eigenbasis: sum_i (r_i^2 / d_i) where r_i = (V^T @ residuals)_i
        V = self.residual_eigvecs
        d = self.residual_eigvals
        r_proj = V.T @ residuals.T  # (param_dim, batch)
        mahal = (r_proj**2 / d.unsqueeze(1)).sum(dim=0)  # (batch,)

        return -0.5 * (self.param_dim * np.log(2 * np.pi) + log_det.detach() + mahal)

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
        mean = self.forward_mean(conditions)
        d = self.residual_eigvals  # posterior covariance eigenvalues
        V = self.residual_eigvecs

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
        """Sample from the posterior distribution (gamma -> infinity equivalent)."""
        mean = self.forward_mean(conditions)
        V = self.residual_eigvecs
        d = self.residual_eigvals

        # Sample from N(mean, residual_cov)
        eps = torch.randn_like(mean)
        return mean + (V @ (torch.sqrt(d).unsqueeze(1) * eps.T)).T


# ==================== SNPE_gaussian Implementation ====================


class SNPE_gaussian(StepwiseEstimator):
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
            theta_key=theta_key,
            condition_keys=condition_keys,
        )

        self.config = config

        # Device setup
        self.device = self._setup_device(config.device)

        # Embedding network
        embedding_config = OmegaConf.to_container(config.network.embedding, resolve=True)
        self._embedding = instantiate_embedding(embedding_config).to(self.device)

        # Posterior network (initialized lazily)
        self._posterior = None
        self._best_posterior = None
        self._best_embedding = None
        self._init_parameters = None

        # Best loss tracking
        self.best_val_loss = float("inf")

        # Optimizer/scheduler (initialized lazily)
        self._optimizer = None
        self._scheduler = None

        # Extended history for tracking
        self.history.update({
            "theta_mins": [],
            "theta_maxs": [],
        })

    def _setup_device(self, device: Optional[str]) -> torch.device:
        """Setup compute device."""
        if device:
            return torch.device(device)
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        debug(f"Auto-detected device: {dev}")
        return dev

    # ==================== Network Initialization ====================

    def _initialize_networks(self, theta: torch.Tensor, conditions: Dict) -> None:
        """Initialize posterior network and optimizer."""
        self._init_parameters = [theta, conditions]
        debug("Initializing networks...")
        debug(f"GPU available: {torch.cuda.is_available()}")

        cfg_net = self.config.network
        cfg_opt = self.config.optimizer

        # Embed conditions to get embedding dimension
        conditions_device = {k: v.to(self.device) for k, v in conditions.items()}
        s = self._embed(conditions_device, train=False).detach()
        theta_device = theta.to(self.device)

        param_dim = theta_device.shape[1]
        condition_dim = s.shape[1]

        # Create posterior network
        self._posterior = GaussianPosterior(
            param_dim=param_dim,
            condition_dim=condition_dim,
            hidden_dim=cfg_net.hidden_dim,
            num_layers=cfg_net.num_layers,
            momentum=cfg_net.momentum,
            min_var=cfg_net.min_var,
            eig_update_freq=cfg_net.eig_update_freq,
        ).to(self.device)

        # Best-fit copy
        self._best_posterior = GaussianPosterior(
            param_dim=param_dim,
            condition_dim=condition_dim,
            hidden_dim=cfg_net.hidden_dim,
            num_layers=cfg_net.num_layers,
            momentum=cfg_net.momentum,
            min_var=cfg_net.min_var,
            eig_update_freq=cfg_net.eig_update_freq,
        ).to(self.device)
        self._best_posterior.load_state_dict(self._posterior.state_dict())

        self._best_embedding = copy.deepcopy(self._embedding)

        # Optimizer and scheduler
        parameters = list(self._posterior.net.parameters()) + list(self._embedding.parameters())
        self._optimizer = AdamW(parameters, lr=cfg_opt.lr)
        self._scheduler = ReduceLROnPlateau(
            self._optimizer,
            mode="min",
            factor=cfg_opt.lr_decay_factor,
            patience=cfg_opt.scheduler_patience,
        )

        self.networks_initialized = True
        debug("Networks initialized.")

    # ==================== Train/Val Steps ====================

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
        """SNPE_gaussian training step with gradient update and optional sample discarding."""
        ids, theta, theta_logprob, conditions, theta_device, conditions_device = \
            self._unpack_batch(batch, "train")

        # Initialize networks on first batch
        if not self.networks_initialized:
            self._initialize_networks(theta, conditions)

        # Embed conditions
        s = self._embed(conditions_device, train=True)

        # Track theta ranges
        with torch.no_grad():
            self.history["theta_mins"].append(theta.min(dim=0).values.cpu().numpy())
            self.history["theta_maxs"].append(theta.max(dim=0).values.cpu().numpy())

        # Update running statistics
        self._posterior.update_stats(theta_device, s.detach())

        # Forward and backward pass
        self._optimizer.zero_grad()
        self._posterior.train()
        log_prob = self._posterior.log_prob(theta_device, s)
        loss = -log_prob.mean()
        loss.backward()
        self._optimizer.step()

        # Update residual covariance
        self._posterior.update_residual_cov(theta_device, s.detach())

        # Discard samples based on log-likelihood ratio
        if self.config.inference.discard_samples:
            discard_mask = self._compute_discard_mask(theta, theta_logprob, conditions_device)
            batch.discard(discard_mask)

        return {"loss": loss.item()}

    def val_step(self, batch) -> Dict[str, float]:
        """SNPE_gaussian validation step without gradient computation."""
        _, theta, theta_logprob, conditions, theta_device, conditions_device = \
            self._unpack_batch(batch, "val")

        # Embed conditions (eval mode)
        s = self._embed(conditions_device, train=False)

        # Compute loss without gradients
        with torch.no_grad():
            self._posterior.eval()
            log_prob = self._posterior.log_prob(theta_device, s)
            loss = -log_prob.mean()

        return {"loss": loss.item()}

    def on_epoch_end(self, epoch: int, val_metrics: Dict[str, float]) -> None:
        """Update best weights and scheduler."""
        val_loss = val_metrics.get("loss", float("inf"))

        # Update best posterior
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self._update_best_weights()
            log({"checkpoint:posterior": epoch})

        # LR scheduler step
        self._scheduler.step(val_loss)
        log({"lr": self._optimizer.param_groups[0]["lr"]})

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

        # Use best model if available and configured
        use_best = cfg_inf.use_best_model_during_inference and self._best_posterior is not None
        if use_best:
            posterior = self._best_posterior
            s = self._embed(conditions, train=False, use_best_fit=True)
        else:
            posterior = self._posterior
            s = self._embed(conditions, train=False)

        s = s.expand(num_samples, *s.shape[1:])

        with torch.no_grad():
            posterior.eval()
            samples = posterior.sample_posterior(s)
            logprob = posterior.log_prob(samples, s)

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

        # Use best model if available and configured
        use_best = cfg_inf.use_best_model_during_inference and self._best_posterior is not None
        if use_best:
            posterior = self._best_posterior
            s = self._embed(conditions, train=False, use_best_fit=True)
        else:
            posterior = self._posterior
            s = self._embed(conditions, train=False)

        s = s.expand(num_samples, *s.shape[1:])

        with torch.no_grad():
            posterior.eval()
            samples = posterior.sample(s, gamma=cfg_inf.gamma)
            logprob = posterior.log_prob(samples, s)

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

        torch.save(self._best_posterior.state_dict(), node_dir / "posterior.pth")
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

        if self._best_embedding is not None:
            torch.save(self._best_embedding.state_dict(), node_dir / "embedding.pth")

    def load(self, node_dir: Path) -> None:
        """Load SNPE_gaussian state."""
        debug(f"Loading: {node_dir}")
        init_parameters = torch.load(node_dir / "init_parameters.pth")
        self._initialize_networks(init_parameters[0], init_parameters[1])

        self._best_posterior.load_state_dict(
            torch.load(node_dir / "posterior.pth")
        )

        if (node_dir / "embedding.pth").exists() and self._best_embedding is not None:
            self._best_embedding.load_state_dict(torch.load(node_dir / "embedding.pth"))

    # ==================== Private Helpers ====================

    def _embed(self, conditions: Dict, train: bool = True, use_best_fit: bool = False):
        """Run conditions through embedding network."""
        embedding = (
            self._best_embedding
            if use_best_fit and self._best_embedding is not None
            else self._embedding
        )
        embedding.train() if train else embedding.eval()
        return embedding(conditions)

    def _update_best_weights(self) -> None:
        """Copy current network weights to best-fit checkpoint."""
        self._best_posterior.load_state_dict(
            {k: v.clone() for k, v in self._posterior.state_dict().items()}
        )
        self._best_embedding.load_state_dict(
            {k: v.clone() for k, v in self._embedding.state_dict().items()}
        )

    def _compute_discard_mask(
        self, theta: torch.Tensor, theta_logprob: torch.Tensor, conditions: Dict
    ):
        """Compute boolean mask of samples to discard based on log-likelihood ratio."""
        cfg_inf = self.config.inference

        theta_device = theta.to(self.device, dtype=torch.float32)
        s = self._embed(conditions, train=False, use_best_fit=True)

        theta_device = theta_device.expand(len(theta), *theta_device.shape[1:]) if theta_device.shape[0] == 1 else theta_device
        s = s.expand(len(theta), *s.shape[1:]) if s.shape[0] == 1 else s

        self._posterior.eval()
        log_prob = self._posterior.log_prob(theta_device, s).cpu()
        log_ratio = log_prob - theta_logprob.cpu()
        return log_ratio < cfg_inf.log_ratio_threshold
