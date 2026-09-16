"""Full-covariance Gaussian estimator for TransformedPrior simulators."""

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from falcon.priors.product import TransformedPrior
from falcon.estimators.networks import build_mlp
from falcon.estimators.torch_model import NetworkGroup, TorchModel, to_numpy_tree
from falcon.core.logger import log, debug


# ==================== _GaussianPosterior Module ====================


class _GaussianPosterior(nn.Module):
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

        # Input statistics (conditions) - diagonal whitening; float64 for precision
        self.register_buffer("_input_mean", torch.zeros(condition_dim, dtype=torch.float64))
        self.register_buffer("_input_std", torch.ones(condition_dim, dtype=torch.float64))

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
        for name in ('_input_mean', '_input_std',
                     '_output_mean', '_output_std', '_residual_cov',
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

        Result dtype matches the parameter-space buffers (i.e. the precision of the
        training parameters), not the conditions dtype, which may be downcast to
        float32 by the embedding layer.

        Args:
            conditions: Condition tensor of shape (batch, condition_dim)
            gamma: Tempering parameter. None = untempered, <1 = widened.
        """
        out_dtype = self._output_mean.dtype
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

        Whitening is done in float64 (statistics precision). The whitened value
        is cast to the MLP's dtype (float32) before the forward pass, then the
        MLP output is upcast to parameter-space dtype (float64) before de-whitening.
        """
        c = conditions.to(self._input_mean.dtype)
        x_norm = (c - self._input_mean.detach()) / self._input_std.detach()
        r = self.net(x_norm.to(next(self.net.parameters()).dtype))
        r = r.to(self._output_mean.dtype)
        return self._output_mean.detach() + self._output_std.detach() * r

    def _update_stats(self, theta: torch.Tensor, conditions: torch.Tensor) -> None:
        """Update running statistics using EMA.

        Both input and output buffers are float64; conditions and theta are cast
        accordingly before computing statistics.
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


class GaussianFullCov(TorchModel):
    """Full-covariance Gaussian posterior estimator for TransformedPrior simulators.

    Works in the standard-normal latent space; samples are mapped back to
    parameter space after generation. Training runs in rounds of epochs on
    fixed data; see ``RoundTrainer``.

    Args:
        max_epochs: Maximum epochs per round.
        lr: Learning rate; reset at the start of every round.
        gamma: Proposal tempering coefficient.
        embedding: Embedding config dict or ``None``.
        device: Device string; auto-detected if ``None``.
        batch_size: Mini-batch size.
        patience_epochs: End the round once the best validation loss is this
            many epochs old (checked at validations).
        val_every_epochs: Validate after every N-th epoch (and at the last epoch).
        max_rounds: Maximum number of rounds (``None`` = unlimited).
        patience_rounds: Stop training after this many rejected rounds in a row.
        prior_rounds: Rounds that simulate from the prior before switching to
            the learned proposal. The first round always uses the prior.
        cache_on_device: Cache training data on the estimator's device.
        max_cache_samples: Cap on cached training samples (0 = all).
        hidden_dim: MLP hidden layer width.
        num_layers: MLP depth.
        momentum: EMA momentum for running statistics.
        min_var: Minimum variance for numerical stability.
        eig_update_freq: Eigendecomposition update frequency.
        betas: AdamW beta coefficients.
        lr_decay_factor: LR decay factor (1.0 = no decay).
        lr_patience_epochs: Epochs without validation improvement before LR decay.
        discard_samples: Run a discard sweep after every accepted round.
        log_ratio_threshold: Log-ratio cutoff for discarding.
    """

    def __init__(
        self,
        *,
        # Most commonly changed
        max_epochs: int = 100,
        lr: float = 1e-2,
        gamma: float = 0.5,
        embedding=None,
        device=None,
        # Training loop
        batch_size: int = 128,
        patience_epochs: int = 16,
        val_every_epochs: int = 1,
        max_rounds: Optional[int] = None,
        patience_rounds: int = 10,
        prior_rounds: int = 0,
        cache_on_device: bool = False,
        max_cache_samples: int = 0,
        # Network architecture
        hidden_dim: int = 128,
        num_layers: int = 3,
        momentum: float = 0.01,
        min_var: float = 1e-20,
        eig_update_freq: int = 1,
        # Optimizer
        betas: tuple = (0.9, 0.9),
        lr_decay_factor: float = 1.0,
        lr_patience_epochs: int = 8,
        # Inference / sampling
        discard_samples: bool = False,
        log_ratio_threshold: float = -20.0,
    ):
        self.max_epochs = max_epochs
        self.lr = lr
        self.gamma = gamma
        self.embedding = embedding
        self.device = device
        self.batch_size = batch_size
        self.patience_epochs = patience_epochs
        self.val_every_epochs = val_every_epochs
        self.max_rounds = max_rounds
        self.patience_rounds = patience_rounds
        self.prior_rounds = prior_rounds
        self.cache_on_device = cache_on_device
        self.max_cache_samples = max_cache_samples
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.momentum = momentum
        self.min_var = min_var
        self.eig_update_freq = eig_update_freq
        self.betas = betas
        self.lr_decay_factor = lr_decay_factor
        self.lr_patience_epochs = lr_patience_epochs
        self.discard_samples = discard_samples
        self.log_ratio_threshold = log_ratio_threshold

    def setup(
        self,
        simulator_instance,
        theta_key: Optional[str] = None,
        condition_keys: Optional[List[str]] = None,
    ):
        if not isinstance(simulator_instance, TransformedPrior):
            raise TypeError(
                f"GaussianFullCov requires a TransformedPrior (e.g., Product), "
                f"got {type(simulator_instance).__name__}."
            )

        super().setup(simulator_instance, theta_key, condition_keys)

        self._proposal_gamma = self.gamma
        self._posterior_gamma = (
            (1.0 + self.gamma) / self.gamma
            if self.gamma is not None else None
        )

        self._model: Optional[nn.Module] = None
        self._optimizer = None
        self._scheduler = None

    # ==================== Optimizer ====================

    def _build_optimizer(self):
        self._optimizer = AdamW(self._model.parameters(), lr=self.lr, betas=self.betas)
        self._build_scheduler()

    def _build_scheduler(self):
        # Stepped once per validation, so the patience is converted from epochs
        self._scheduler = (
            ReduceLROnPlateau(
                self._optimizer, mode="min",
                factor=self.lr_decay_factor,
                patience=-(-self.lr_patience_epochs // self.val_every_epochs),
            )
            if self.lr_decay_factor < 1.0 else None
        )

    # ==================== Model Building ====================

    def init_from_batch(self, batch):
        return {
            "theta": self._to_array(batch[f"{self.theta_key}.value"]),
            "conditions": {
                k: self._to_array(batch[f"{k}.value"])
                for k in self.condition_keys if f"{k}.value" in batch
            },
        }

    def build(self, init_tree) -> None:
        from falcon.estimators.embedded_posterior import EmbeddedPosterior
        from falcon.embeddings import instantiate_embedding

        theta = self._to_tensor(init_tree["theta"])
        conditions = {k: self._to_tensor(v, self.device) for k, v in init_tree["conditions"].items()}
        theta_latent = self.simulator_instance.inverse(theta, mode="standard_normal")

        embedding = instantiate_embedding(self.embedding).to(self.device)
        embedding.eval()
        with torch.no_grad():
            embedded = embedding(conditions)

        posterior = _GaussianPosterior(
            param_dim=theta_latent.shape[1],
            condition_dim=embedded.shape[1],
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            momentum=self.momentum,
            min_var=self.min_var,
            eig_update_freq=self.eig_update_freq,
        ).to(self.device)

        self._model = EmbeddedPosterior(embedding, posterior)
        self._set_groups({
            "model": NetworkGroup({"model": self._model}, "loss", embedding=self._model.embedding),
        })
        debug(f"GaussianFullCov model built: param_dim={theta_latent.shape[1]}")

    # ==================== Loss ====================

    def _unpack_batch(self, batch):
        theta = self._to_tensor(batch[f"{self.theta_key}.value"], self.device)
        theta_logprob = self._to_tensor(batch[f"{self.theta_key}.log_prob"])
        conditions = {
            k: self._to_tensor(batch[f"{k}.value"], self.device)
            for k in self.condition_keys if f"{k}.value" in batch
        }
        theta_latent = self.simulator_instance.inverse(theta, mode="standard_normal")
        return theta_latent, theta_logprob, conditions

    def _compute_loss(self, batch, train: bool):
        theta_latent, _, conditions = self._unpack_batch(batch)
        self._model.train(train)
        s = self._summary("model", conditions, train=train)
        loss = self._model.posterior.loss(theta_latent, s)
        return loss, {"loss": loss.item()}

    # ==================== Training ====================

    def train_step(self, batch) -> Dict[str, float]:
        if self._optimizer is None:
            self._build_optimizer()
        self._optimizer.zero_grad()
        loss, metrics = self._compute_loss(batch, train=True)
        loss.backward()
        self._optimizer.step()
        return metrics

    def val_step(self, batch) -> Dict[str, float]:
        _, metrics = self._compute_loss(batch, train=False)
        return metrics

    def discard_test(self, batch):
        theta_latent, theta_logprob, conditions = self._unpack_batch(batch)
        self._model.eval()
        s = self._summary("model", conditions)
        log_prob = self._model.posterior.log_prob(theta_latent, s).cpu()
        return (log_prob - theta_logprob.cpu()) < self.log_ratio_threshold

    def on_round_start(self) -> None:
        if self._optimizer is None:
            return
        for group in self._optimizer.param_groups:
            group["lr"] = self.lr
        self._build_scheduler()

    def on_validation_end(self, epoch: int, val_metrics: Dict[str, float]) -> Optional[Dict[str, float]]:
        val_loss = val_metrics.get("loss", float("inf"))

        if self._scheduler is not None:
            self._scheduler.step(val_loss)
        lr = self._optimizer.param_groups[0]["lr"]
        log({"lr": lr})

        extra = {"lr": lr}
        posterior = self._model.posterior
        if hasattr(posterior, "_output_std"):
            extra["theta_std"] = posterior._output_std.mean().item()
        if hasattr(posterior, "_residual_eigvals"):
            extra["eigvals_mean"] = posterior._residual_eigvals.mean().item()
        return extra

    # ==================== Sampling ====================

    def _sample_prior(self, num_samples: int) -> dict:
        samples = self.simulator_instance.simulate_batch(num_samples)
        return {"value": samples, "log_prob": np.zeros(num_samples)}

    def _sample(self, num_samples: int, conditions, mode: str) -> dict:
        gamma = self._proposal_gamma if mode == "proposal" else self._posterior_gamma
        conditions_device = {
            k: self._to_tensor(v, self.device).expand(num_samples, *v.shape[1:])
            for k, v in conditions.items()
        }

        self._model.eval()
        samples_latent = self._model.sample(conditions_device, gamma=gamma)
        log_prob = self._model.log_prob(samples_latent, conditions_device)
        samples = self.simulator_instance.forward(samples_latent, mode="standard_normal")
        result = {"value": samples.cpu().numpy(), "log_prob": log_prob.cpu().numpy()}

        if mode == "proposal":
            log({
                "sample_proposal:mean": result["value"].mean(),
                "sample_proposal:std": result["value"].std(),
                "sample_proposal:logprob": result["log_prob"].mean(),
            })
        return result

    # ==================== Legacy checkpoints ====================

    def load_legacy(self, node_dir: Path):
        """Checkpoint files written before ``best_state.npz`` existed."""
        if not (node_dir / "model.pth").exists():
            return None
        init = self._legacy_load(node_dir / "init_tensors.pth")
        return {
            "meta": self._legacy_meta(node_dir),
            "init": {
                "theta": self._to_array(init["theta"]),
                "conditions": to_numpy_tree(init["conditions"]),
            },
            "groups": {"model": {"model": to_numpy_tree(self._legacy_load(node_dir / "model.pth"))}},
        }
