"""Sequential Neural Posterior Estimation (SNPE-A) implementation."""

import copy
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from falcon.core.utils import RVBatch
from falcon.core.logger import log, debug, info, warning, error
from falcon.contrib.flow import Flow
from falcon.contrib.stepwise_estimator import StepwiseEstimator, TrainingLoopConfig
from falcon.contrib.torch_embedding import instantiate_embedding


# ==================== Configuration Dataclasses ====================


@dataclass
class NetworkConfig:
    """Neural network architecture parameters."""

    net_type: str = "zuko_nice"
    theta_norm: bool = True
    norm_momentum: float = 1e-2
    adaptive_momentum: bool = False
    use_log_update: bool = False
    embedding: Optional[Any] = None


@dataclass
class OptimizerConfig:
    """Optimizer parameters (training-time)."""

    lr: float = 1e-2
    lr_decay_factor: float = 0.1
    scheduler_patience: int = 8


@dataclass
class InferenceConfig:
    """Inference and sampling parameters."""

    gamma: float = 0.5
    discard_samples: bool = True
    log_ratio_threshold: float = -20.0
    sample_reference_posterior: bool = False
    use_best_models_during_inference: bool = True
    # Importance sampling parameters
    num_proposals: int = 256
    reference_samples: int = 128
    hypercube_bound: float = 2.0
    out_of_bounds_penalty: float = 100.0
    nan_replacement: float = -100.0


@dataclass
class SNPEConfig:
    """Top-level SNPE_A configuration."""

    loop: TrainingLoopConfig = field(default_factory=TrainingLoopConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    device: Optional[str] = None


# ==================== SNPE_A Implementation ====================


class SNPE_A(StepwiseEstimator):
    """
    Sequential Neural Posterior Estimation (SNPE-A).

    Implementation-specific features:
    - Dual flow architecture (conditional + marginal)
    - Parameter space normalization via hypercube mapping
    - Importance sampling for posterior/proposal
    """

    def __init__(
        self,
        simulator_instance,
        theta_key: Optional[str] = None,
        condition_keys: Optional[List[str]] = None,
        config: Optional[dict] = None,
    ):
        """
        Initialize SNPE_A estimator.

        Args:
            simulator_instance: Prior/simulator instance
            theta_key: Key for theta in batch data
            condition_keys: Keys for condition data in batch
            config: Configuration dict with loop, network, optimizer, inference sections
        """
        # Merge user config with defaults using OmegaConf structured config
        schema = OmegaConf.structured(SNPEConfig)
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
        # Convert to plain dict for instantiate_embedding which uses isinstance(x, dict)
        embedding_config = OmegaConf.to_container(config.network.embedding, resolve=True)
        self._embedding = instantiate_embedding(embedding_config).to(self.device)

        # Flow networks (initialized lazily)
        self._conditional_flow = None
        self._marginal_flow = None
        self._best_conditional_flow = None
        self._best_marginal_flow = None
        self._best_embedding = None
        self._init_parameters = None

        # Best loss tracking
        self.best_conditional_flow_val_loss = float("inf")
        self.best_marginal_flow_val_loss = float("inf")

        # Optimizer/scheduler (initialized lazily)
        self._optimizer = None
        self._scheduler = None

        # Extended history for SNPE-specific tracking
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
        """Initialize flow networks and optimizer."""
        self._init_parameters = [theta, conditions]
        debug("Initializing networks...")
        debug(f"GPU available: {torch.cuda.is_available()}")

        cfg_net = self.config.network
        cfg_opt = self.config.optimizer

        # Embed conditions to get embedding dimension
        conditions_device = {k: v.to(self.device) for k, v in conditions.items()}
        s = self._embed(conditions_device, train=False).detach()
        theta_device = theta.to(self.device)

        # Create flow networks
        self._conditional_flow = self._create_flow(theta_device, s, is_conditional=True)
        self._conditional_flow.to(self.device)

        self._marginal_flow = self._create_flow(theta_device, s, is_conditional=False)
        self._marginal_flow.to(self.device)

        # Best-fit copies
        self._best_conditional_flow = self._create_flow(theta_device, s, is_conditional=True)
        self._best_conditional_flow.to(self.device)
        self._best_conditional_flow.load_state_dict(self._conditional_flow.state_dict())

        self._best_marginal_flow = self._create_flow(theta_device, s, is_conditional=False)
        self._best_marginal_flow.to(self.device)
        self._best_marginal_flow.load_state_dict(self._marginal_flow.state_dict())

        self._best_embedding = copy.deepcopy(self._embedding)

        # Optimizer and scheduler
        parameters = (
            list(self._conditional_flow.parameters())
            + list(self._marginal_flow.parameters())
            + list(self._embedding.parameters())
        )
        self._optimizer = AdamW(parameters, lr=cfg_opt.lr)
        self._scheduler = ReduceLROnPlateau(
            self._optimizer,
            mode="min",
            factor=cfg_opt.lr_decay_factor,
            patience=cfg_opt.scheduler_patience,
        )

        self.networks_initialized = True
        debug("Networks initialized.")

    def _create_flow(self, theta, s, is_conditional=True):
        """Create a Flow network with current config."""
        cfg = self.config.network
        return Flow(
            theta,
            s if is_conditional else s * 0,
            theta_norm=cfg.theta_norm,
            norm_momentum=cfg.norm_momentum,
            net_type=cfg.net_type,
            use_log_update=cfg.use_log_update,
            adaptive_momentum=cfg.adaptive_momentum,
        )

    # ==================== Train/Val Steps ====================

    def _unpack_batch(self, batch, phase: str):
        """Unpack batch data and convert to tensors.

        Args:
            batch: Batch object with theta, logprob, and conditions
            phase: "train" or "val" for logging and history

        Returns:
            Tuple of (ids, theta, theta_logprob, conditions, u, u_device, conditions_device)
        """
        ids = batch._ids
        theta = self._to_tensor(batch[self.theta_key])
        theta_logprob = self._to_tensor(batch[f"{self.theta_key}.logprob"])
        conditions = {
            k: self._to_tensor(batch[k]) for k in self.condition_keys if k in batch
        }

        # Record IDs for history
        ts = time.time()
        self.history[f"{phase}_ids"].extend((ts, id) for id in ids.tolist())

        log({f"{phase}:theta_logprob_min": theta_logprob.min().item()})
        log({f"{phase}:theta_logprob_max": theta_logprob.max().item()})

        # Transform to hypercube space
        u = self.simulator_instance.inverse(theta)

        # Move to device
        conditions_device = {k: v.to(self.device) for k, v in conditions.items()}
        u_device = u.to(self.device)

        return ids, theta, theta_logprob, conditions, u, u_device, conditions_device

    def _compute_flow_losses(self, u_device, s, train: bool):
        """Compute conditional and marginal flow losses.

        Args:
            u_device: Transformed parameters on device
            s: Embedded conditions
            train: Whether in training mode

        Returns:
            Tuple of (loss_cond, loss_marg) tensors
        """
        if train:
            self._conditional_flow.train()
            self._marginal_flow.train()
        else:
            self._conditional_flow.eval()
            self._marginal_flow.eval()

        loss_cond = self._conditional_flow.loss(u_device, s).mean()
        # Zero out conditions for marginal flow (detach in train mode to avoid backprop)
        s_marginal = s.detach() * 0 if train else s * 0
        loss_marg = self._marginal_flow.loss(u_device, s_marginal).mean()

        return loss_cond, loss_marg

    def train_step(self, batch) -> Dict[str, float]:
        """SNPE-A training step with gradient update and optional sample discarding."""
        ids, theta, theta_logprob, conditions, u, u_device, conditions_device = \
            self._unpack_batch(batch, "train")

        # Initialize networks on first batch
        if not self.networks_initialized:
            self._initialize_networks(u, conditions)

        # Embed conditions
        s = self._embed(conditions_device, train=True)

        # Track theta ranges
        with torch.no_grad():
            self.history["theta_mins"].append(theta.min(dim=0).values.cpu().numpy())
            self.history["theta_maxs"].append(theta.max(dim=0).values.cpu().numpy())

        # Forward and backward pass
        self._optimizer.zero_grad()
        loss_cond, loss_marg = self._compute_flow_losses(u_device, s, train=True)
        (loss_cond + loss_marg).backward()
        self._optimizer.step()

        # Discard samples based on log-likelihood ratio
        if self.config.inference.discard_samples:
            discard_mask = self._compute_discard_mask(theta, theta_logprob, conditions_device)
            batch.discard(discard_mask)

        return {"loss": loss_cond.item(), "loss_aux": loss_marg.item()}

    def val_step(self, batch) -> Dict[str, float]:
        """SNPE-A validation step without gradient computation."""
        _, theta, theta_logprob, conditions, u, u_device, conditions_device = \
            self._unpack_batch(batch, "val")

        # Embed conditions (eval mode)
        s = self._embed(conditions_device, train=False)

        # Compute losses without gradients
        with torch.no_grad():
            loss_cond, loss_marg = self._compute_flow_losses(u_device, s, train=False)

        return {"loss": loss_cond.item(), "loss_aux": loss_marg.item()}

    def on_epoch_end(self, epoch: int, val_metrics: Dict[str, float]) -> Optional[Dict[str, float]]:
        """Update best weights and scheduler."""
        val_loss = val_metrics.get("loss", float("inf"))
        val_aux_loss = val_metrics.get("loss_aux", float("inf"))

        # Update best conditional flow
        if val_loss < self.best_conditional_flow_val_loss:
            self.best_conditional_flow_val_loss = val_loss
            self._update_best_weights("conditional")
            log({"checkpoint:conditional": epoch})

        # Update best marginal flow
        if val_aux_loss < self.best_marginal_flow_val_loss:
            self.best_marginal_flow_val_loss = val_aux_loss
            self._update_best_weights("marginal")
            log({"checkpoint:marginal": epoch})

        # LR scheduler step
        self._scheduler.step(val_loss)
        lr = self._optimizer.param_groups[0]["lr"]
        log({"lr": lr})

        return {"lr": lr}

    # ==================== Sampling Methods ====================

    def sample_prior(self, num_samples: int, conditions: Optional[Dict] = None) -> RVBatch:
        """Sample from the prior distribution."""
        if conditions:
            raise ValueError("Conditions are not supported for sample_prior.")
        samples = self.simulator_instance.simulate_batch(num_samples)
        # Log probability for uniform prior over hypercube [-bound, bound]^d
        bound = self.config.inference.hypercube_bound
        logprob = np.ones(num_samples) * (-np.log(2 * bound) ** self.param_dim)
        return RVBatch(samples, logprob=logprob)

    def sample_posterior(
        self,
        num_samples: int,
        conditions: Optional[Dict] = None,
    ) -> RVBatch:
        """Sample from the posterior distribution q(theta|x)."""
        # Fall back to prior if networks not yet initialized (training hasn't started)
        if not self.networks_initialized:
            return self.sample_prior(num_samples)

        samples, logprob = self._importance_sample(num_samples, mode="posterior", conditions=conditions or {})
        return RVBatch(samples.numpy(), logprob=logprob.numpy())

    def sample_proposal(
        self,
        num_samples: int,
        conditions: Optional[Dict] = None,
    ) -> RVBatch:
        """Sample from the widened proposal distribution for adaptive resampling."""
        # Fall back to prior if networks not yet initialized (training hasn't started)
        if not self.networks_initialized:
            return self.sample_prior(num_samples)

        cfg_inf = self.config.inference
        conditions = conditions or {}

        if cfg_inf.sample_reference_posterior:
            post_samples, _ = self._importance_sample(cfg_inf.reference_samples, mode="posterior", conditions=conditions)
            mean, std = post_samples.mean(dim=0).cpu(), post_samples.std(dim=0).cpu()
            log({f"sample_proposal:posterior_mean_{i}": mean[i].item() for i in range(len(mean))})
            log({f"sample_proposal:posterior_std_{i}": std[i].item() for i in range(len(std))})

        samples, logprob = self._importance_sample(num_samples, mode="proposal", conditions=conditions)
        log({
            "sample_proposal:mean": samples.mean().item(),
            "sample_proposal:std": samples.std().item(),
            "sample_proposal:logprob": logprob.mean().item(),
        })
        return RVBatch(samples.numpy(), logprob=logprob.numpy())

    def _importance_sample(
        self,
        num_samples: int,
        mode: str = "posterior",
        conditions: Dict = {},
    ):
        """Sample using importance sampling."""
        cfg_inf = self.config.inference

        assert conditions, "Conditions must be provided."
        # Move conditions to device
        conditions = {k: v.to(self.device) for k, v in conditions.items()}

        # Use best models if available and configured, otherwise fall back to current
        use_best = cfg_inf.use_best_models_during_inference and self._best_conditional_flow is not None
        if use_best:
            conditional_net = self._best_conditional_flow
            marginal_net = self._best_marginal_flow
            s = self._embed(conditions, train=False, use_best_fit=True)
        else:
            conditional_net = self._conditional_flow
            marginal_net = self._marginal_flow
            s = self._embed(conditions, train=False)

        s = s.expand(num_samples, *s.shape[1:])

        # Generate proposals from conditional flow
        conditional_net.eval()
        samples_proposals = conditional_net.sample(cfg_inf.num_proposals, s).detach()

        log({
            "importance_sample:proposal_mean": samples_proposals.mean().item(),
            "importance_sample:proposal_std": samples_proposals.std().item(),
        })

        # Compute log probs
        log_prob_cond = conditional_net.log_prob(samples_proposals, s)
        marginal_net.eval()
        log_prob_marg = marginal_net.log_prob(samples_proposals, s * 0)

        # Mask samples outside hypercube bounds
        bound = cfg_inf.hypercube_bound
        mask = (samples_proposals < -bound) | (samples_proposals > bound)
        mask = mask.any(dim=-1).float() * cfg_inf.out_of_bounds_penalty

        # Compute importance weights
        if mode == "proposal":
            log_weights = -1.0 / (1.0 + cfg_inf.gamma) * log_prob_cond - mask
        else:  # "posterior" - reweight by marginal
            log_weights = -log_prob_marg - mask

        nan_val = cfg_inf.nan_replacement
        log_weights = torch.nan_to_num(log_weights, nan=nan_val, neginf=nan_val)
        log_weights = log_weights - torch.logsumexp(log_weights, dim=0, keepdim=True)
        weights = torch.exp(log_weights)

        # Effective sample size
        n_eff = 1 / (weights**2).sum(dim=0).cpu().detach().numpy()
        log({"importance_sample:n_eff_min": n_eff.min()})
        log({"importance_sample:n_eff_max": n_eff.max()})

        # Resample
        idx = torch.multinomial(weights.T, 1, replacement=True).squeeze(-1)
        samples = samples_proposals[idx, torch.arange(num_samples), :]
        samples = self.simulator_instance.forward(samples).cpu()
        logprob = log_prob_cond[idx, torch.arange(num_samples)].cpu()

        return samples, logprob.detach()

    # ==================== Save/Load ====================

    def save(self, node_dir: Path) -> None:
        """Save SNPE-A state."""
        debug(f"Saving: {node_dir}")
        if not self.networks_initialized:
            raise RuntimeError("Networks not initialized.")

        torch.save(self._best_conditional_flow.state_dict(), node_dir / "conditional_flow.pth")
        torch.save(self._best_marginal_flow.state_dict(), node_dir / "marginal_flow.pth")
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
        """Load SNPE-A state."""
        debug(f"Loading: {node_dir}")
        init_parameters = torch.load(node_dir / "init_parameters.pth")
        self._initialize_networks(init_parameters[0], init_parameters[1])

        self._best_conditional_flow.load_state_dict(
            torch.load(node_dir / "conditional_flow.pth")
        )
        self._best_marginal_flow.load_state_dict(
            torch.load(node_dir / "marginal_flow.pth")
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

    def _update_best_weights(self, network_type: str) -> None:
        """Copy current network weights to best-fit checkpoint."""
        if network_type == "conditional":
            self._best_conditional_flow.load_state_dict(
                {k: v.clone() for k, v in self._conditional_flow.state_dict().items()}
            )
            self._best_embedding.load_state_dict(
                {k: v.clone() for k, v in self._embedding.state_dict().items()}
            )
        else:
            self._best_marginal_flow.load_state_dict(
                {k: v.clone() for k, v in self._marginal_flow.state_dict().items()}
            )

    def _compute_discard_mask(
        self, theta: torch.Tensor, theta_logprob: torch.Tensor, conditions: Dict
    ):
        """Compute boolean mask of samples to discard based on log-likelihood ratio."""
        cfg_inf = self.config.inference

        u = self.simulator_instance.inverse(theta)
        s = self._embed(conditions, train=False, use_best_fit=True)

        u = u.expand(len(theta), *u.shape[1:]) if u.shape[0] == 1 else u
        s = s.expand(len(theta), *s.shape[1:]) if s.shape[0] == 1 else s

        u = u.to(self.device)
        self._conditional_flow.eval()
        log_prob = self._conditional_flow.log_prob(u.unsqueeze(0), s).squeeze(0).cpu()
        log_ratio = log_prob - theta_logprob.cpu()
        return log_ratio < cfg_inf.log_ratio_threshold
