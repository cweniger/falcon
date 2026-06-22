"""Flow-based posterior estimation (was SNPE_A)."""

import copy
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from falcon.core.logger import log, debug
from falcon.estimators.flow_density import FlowDensity
from falcon.estimators.stepwise_base import StepwiseEstimator
from falcon.embeddings import instantiate_embedding


class Flow(StepwiseEstimator):
    """Flow-based posterior estimation using a conditional + marginal flow pair.

    Args:
        max_epochs: Maximum training epochs.
        net_type: Flow architecture (``zuko_nice``, ``nsf``, ``maf``, ``zuko_gf``, ...).
        lr: Learning rate.
        gamma: Proposal tempering coefficient.
        embedding: Embedding config dict (with ``_target_`` etc.) or ``None``.
        device: Device string (e.g. ``"cuda:0"``); auto-detected if ``None``.
        batch_size: Mini-batch size.
        early_stop_patience: Epochs without improvement before stopping.
        prior_epochs: Epochs to sample from prior before switching to proposal.
        cache_on_device: Cache training data on the estimator device.
        cache_sync_every: Resync buffer cache every N epochs (0 = every epoch).
        max_cache_samples: Cap on cached training samples (0 = all).
        theta_norm: Normalise parameter space online.
        norm_momentum: EMA momentum for online normalisation.
        adaptive_momentum: Adaptive momentum for normalisation.
        use_log_update: Use log-space normalisation update.
        betas: AdamW beta coefficients.
        lr_decay_factor: LR decay factor for plateau scheduler.
        lr_patience: Plateau patience before LR decay.
        discard_samples: Discard low log-ratio training samples.
        log_ratio_threshold: Log-ratio cutoff for discarding.
        sample_reference_posterior: Sample reference posterior for proposals.
        use_best_models: Use best-checkpoint networks for sampling.
        num_proposals: Importance sampling proposal count.
        reference_samples: Reference posterior sample count.
        hypercube_bound: Hypercube clipping bound for proposals.
        out_of_bounds_penalty: Log-weight penalty for out-of-bounds samples.
        nan_replacement: Replacement for NaN/−∞ log-weights.
    """

    def __init__(
        self,
        *,
        # Most commonly changed
        max_epochs: int = 100,
        net_type: str = "zuko_nice",
        lr: float = 1e-2,
        gamma: float = 0.5,
        embedding=None,
        device: Optional[str] = None,
        # Training loop
        batch_size: int = 128,
        early_stop_patience: int = 16,
        prior_epochs: int = 0,
        cache_on_device: bool = False,
        cache_sync_every: int = 0,
        max_cache_samples: int = 0,
        # Network
        theta_norm: bool = True,
        norm_momentum: float = 1e-2,
        adaptive_momentum: bool = False,
        use_log_update: bool = False,
        # Optimizer
        betas: tuple = (0.9, 0.9),
        lr_decay_factor: float = 0.1,
        lr_patience: int = 8,
        # Inference
        discard_samples: bool = True,
        log_ratio_threshold: float = -20.0,
        sample_reference_posterior: bool = False,
        use_best_models: bool = True,
        num_proposals: int = 256,
        reference_samples: int = 128,
        hypercube_bound: float = 2.0,
        out_of_bounds_penalty: float = 100.0,
        nan_replacement: float = -100.0,
    ):
        self.max_epochs = max_epochs
        self.net_type = net_type
        self.lr = lr
        self.gamma = gamma
        self.embedding = embedding
        self.device = device
        self.batch_size = batch_size
        self.early_stop_patience = early_stop_patience
        self.prior_epochs = prior_epochs
        self.cache_on_device = cache_on_device
        self.cache_sync_every = cache_sync_every
        self.max_cache_samples = max_cache_samples
        self.theta_norm = theta_norm
        self.norm_momentum = norm_momentum
        self.adaptive_momentum = adaptive_momentum
        self.use_log_update = use_log_update
        self.betas = betas
        self.lr_decay_factor = lr_decay_factor
        self.lr_patience = lr_patience
        self.discard_samples = discard_samples
        self.log_ratio_threshold = log_ratio_threshold
        self.sample_reference_posterior = sample_reference_posterior
        self.use_best_models = use_best_models
        self.num_proposals = num_proposals
        self.reference_samples = reference_samples
        self.hypercube_bound = hypercube_bound
        self.out_of_bounds_penalty = out_of_bounds_penalty
        self.nan_replacement = nan_replacement

    def setup(self, simulator_instance, theta_key=None, condition_keys=None):
        super().setup(simulator_instance, theta_key, condition_keys)

        if self.device:
            self.device = torch.device(self.device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            debug(f"Auto-detected device: {self.device}")

        self._embedding = instantiate_embedding(self.embedding).to(self.device)

        self._conditional_flow = None
        self._marginal_flow = None
        self._best_conditional_flow = None
        self._best_marginal_flow = None
        self._best_embedding = None
        self._init_parameters = None

        self.best_conditional_flow_val_loss = float("inf")
        self.best_marginal_flow_val_loss = float("inf")

        self._optimizer = None
        self._scheduler = None

        self.history.update({"theta_mins": [], "theta_maxs": []})

    # ==================== Network Initialization ====================

    def _initialize_networks(self, theta: torch.Tensor, conditions: Dict) -> None:
        debug("Initializing networks...")
        self._init_parameters = [theta, conditions]

        conditions_device = {k: v.to(self.device) for k, v in conditions.items()}
        s = self._embed(conditions_device, train=False).detach()
        theta_device = theta.to(self.device)

        self._conditional_flow = self._create_flow(theta_device, s, is_conditional=True)
        self._conditional_flow.to(self.device)

        self._marginal_flow = self._create_flow(theta_device, s, is_conditional=False)
        self._marginal_flow.to(self.device)

        self._best_conditional_flow = self._create_flow(theta_device, s, is_conditional=True)
        self._best_conditional_flow.to(self.device)
        self._best_conditional_flow.load_state_dict(self._conditional_flow.state_dict())

        self._best_marginal_flow = self._create_flow(theta_device, s, is_conditional=False)
        self._best_marginal_flow.to(self.device)
        self._best_marginal_flow.load_state_dict(self._marginal_flow.state_dict())

        self._best_embedding = copy.deepcopy(self._embedding)

        parameters = (
            list(self._conditional_flow.parameters())
            + list(self._marginal_flow.parameters())
            + list(self._embedding.parameters())
        )
        self._optimizer = AdamW(parameters, lr=self.lr, betas=self.betas)
        self._scheduler = ReduceLROnPlateau(
            self._optimizer,
            mode="min",
            factor=self.lr_decay_factor,
            patience=self.lr_patience,
        )

        self.networks_initialized = True
        debug("Networks initialized.")

    def _create_flow(self, theta, s, is_conditional=True):
        return FlowDensity(
            theta,
            s if is_conditional else s * 0,
            theta_norm=self.theta_norm,
            norm_momentum=self.norm_momentum,
            net_type=self.net_type,
            use_log_update=self.use_log_update,
            adaptive_momentum=self.adaptive_momentum,
        )

    # ==================== Train/Val Steps ====================

    def _unpack_batch(self, batch, phase: str):
        ids = batch._ids
        theta = self._to_tensor(batch[f"{self.theta_key}.value"])
        theta_logprob = self._to_tensor(batch[f"{self.theta_key}.log_prob"])
        conditions = {
            k: self._to_tensor(batch[f"{k}.value"])
            for k in self.condition_keys if f"{k}.value" in batch
        }

        ts = time.time()
        self.history[f"{phase}_ids"].extend((ts, id) for id in ids.tolist())

        log({f"{phase}:theta_logprob_min": theta_logprob.min().item()})
        log({f"{phase}:theta_logprob_max": theta_logprob.max().item()})

        u = self.simulator_instance.inverse(theta)
        conditions_device = {k: v.to(self.device) for k, v in conditions.items()}
        u_device = u.to(self.device)

        return ids, theta, theta_logprob, conditions, u, u_device, conditions_device

    def _compute_flow_losses(self, u_device, s, train: bool):
        if train:
            self._conditional_flow.train()
            self._marginal_flow.train()
        else:
            self._conditional_flow.eval()
            self._marginal_flow.eval()

        loss_cond = self._conditional_flow.loss(u_device, s).mean()
        s_marginal = s.detach() * 0 if train else s * 0
        loss_marg = self._marginal_flow.loss(u_device, s_marginal).mean()

        return loss_cond, loss_marg

    def train_step(self, batch) -> Dict[str, float]:
        ids, theta, theta_logprob, conditions, u, u_device, conditions_device = \
            self._unpack_batch(batch, "train")

        if not self.networks_initialized:
            self._initialize_networks(u, conditions)

        s = self._embed(conditions_device, train=True)

        with torch.no_grad():
            self.history["theta_mins"].append(theta.min(dim=0).values.cpu().numpy())
            self.history["theta_maxs"].append(theta.max(dim=0).values.cpu().numpy())

        self._optimizer.zero_grad()
        loss_cond, loss_marg = self._compute_flow_losses(u_device, s, train=True)
        (loss_cond + loss_marg).backward()
        self._optimizer.step()

        if self.discard_samples:
            discard_mask = self._compute_discard_mask(theta, theta_logprob, conditions_device)
            batch.discard(discard_mask)

        return {"loss": loss_cond.item(), "loss_aux": loss_marg.item()}

    def val_step(self, batch) -> Dict[str, float]:
        _, theta, theta_logprob, conditions, u, u_device, conditions_device = \
            self._unpack_batch(batch, "val")

        s = self._embed(conditions_device, train=False)

        with torch.no_grad():
            loss_cond, loss_marg = self._compute_flow_losses(u_device, s, train=False)

        return {"loss": loss_cond.item(), "loss_aux": loss_marg.item()}

    def on_epoch_end(self, epoch: int, val_metrics: Dict[str, float]) -> Optional[Dict[str, float]]:
        val_loss = val_metrics.get("loss", float("inf"))
        val_aux_loss = val_metrics.get("loss_aux", float("inf"))

        if val_loss < self.best_conditional_flow_val_loss:
            self.best_conditional_flow_val_loss = val_loss
            self._update_best_weights("conditional")
            log({"checkpoint:conditional": epoch})

        if val_aux_loss < self.best_marginal_flow_val_loss:
            self.best_marginal_flow_val_loss = val_aux_loss
            self._update_best_weights("marginal")
            log({"checkpoint:marginal": epoch})

        self._scheduler.step(val_loss)
        lr = self._optimizer.param_groups[0]["lr"]
        log({"lr": lr})

        return {"lr": lr}

    # ==================== Sampling ====================

    def sample_prior(self, num_samples: int, conditions=None) -> dict:
        if conditions:
            raise ValueError("Conditions are not supported for sample_prior.")
        samples = self.simulator_instance.simulate_batch(num_samples)
        log_prob = np.ones(num_samples) * (-np.log(2 * self.hypercube_bound) ** self.param_dim)
        return {'value': samples, 'log_prob': log_prob}

    def sample_posterior(self, num_samples: int, conditions=None) -> dict:
        if not self.networks_initialized:
            return self.sample_prior(num_samples)
        samples, logprob = self._importance_sample(num_samples, mode="posterior", conditions=conditions or {})
        return {'value': samples.numpy(), 'log_prob': logprob.numpy()}

    def sample_proposal(self, num_samples: int, conditions=None) -> dict:
        if self._total_epochs_trained < self.prior_epochs:
            return self.sample_prior(num_samples)
        if not self.networks_initialized:
            return self.sample_prior(num_samples)

        conditions = conditions or {}
        if self.sample_reference_posterior:
            post_samples, _ = self._importance_sample(
                self.reference_samples, mode="posterior", conditions=conditions
            )
            mean, std = post_samples.mean(dim=0).cpu(), post_samples.std(dim=0).cpu()
            log({f"sample_proposal:posterior_mean_{i}": mean[i].item() for i in range(len(mean))})
            log({f"sample_proposal:posterior_std_{i}": std[i].item() for i in range(len(std))})

        samples, logprob = self._importance_sample(num_samples, mode="proposal", conditions=conditions)
        log({
            "sample_proposal:mean": samples.mean().item(),
            "sample_proposal:std": samples.std().item(),
            "sample_proposal:logprob": logprob.mean().item(),
        })
        return {'value': samples.numpy(), 'log_prob': logprob.numpy()}

    def _importance_sample(self, num_samples: int, mode: str = "posterior", conditions: Dict = {}):
        assert conditions, "Conditions must be provided."
        conditions = {k: self._to_tensor(v, self.device) for k, v in conditions.items()}

        use_best = self.use_best_models and self._best_conditional_flow is not None
        if use_best:
            conditional_net = self._best_conditional_flow
            marginal_net = self._best_marginal_flow
            s = self._embed(conditions, train=False, use_best_fit=True)
        else:
            conditional_net = self._conditional_flow
            marginal_net = self._marginal_flow
            s = self._embed(conditions, train=False)

        s = s.expand(num_samples, *s.shape[1:])

        conditional_net.eval()
        samples_proposals = conditional_net.sample(self.num_proposals, s).detach()

        log({
            "importance_sample:proposal_mean": samples_proposals.mean().item(),
            "importance_sample:proposal_std": samples_proposals.std().item(),
        })

        log_prob_cond = conditional_net.log_prob(samples_proposals, s)
        marginal_net.eval()
        log_prob_marg = marginal_net.log_prob(samples_proposals, s * 0)

        mask = (samples_proposals < -self.hypercube_bound) | (samples_proposals > self.hypercube_bound)
        mask = mask.any(dim=-1).float() * self.out_of_bounds_penalty

        if mode == "proposal":
            log_weights = -1.0 / (1.0 + self.gamma) * log_prob_cond - mask
        else:
            log_weights = -log_prob_marg - mask

        log_weights = torch.nan_to_num(log_weights, nan=self.nan_replacement, neginf=self.nan_replacement)
        log_weights = log_weights - torch.logsumexp(log_weights, dim=0, keepdim=True)
        weights = torch.exp(log_weights)

        n_eff = 1 / (weights**2).sum(dim=0).cpu().detach().numpy()
        log({"importance_sample:n_eff_min": n_eff.min()})
        log({"importance_sample:n_eff_max": n_eff.max()})

        idx = torch.multinomial(weights.T, 1, replacement=True).squeeze(-1)
        samples = samples_proposals[idx, torch.arange(num_samples), :]
        samples = self.simulator_instance.forward(samples).cpu()
        logprob = log_prob_cond[idx, torch.arange(num_samples)].cpu()

        return samples, logprob.detach()

    # ==================== Save/Load ====================

    def save(self, node_dir: Path) -> None:
        debug(f"Saving: {node_dir}")
        if not self.networks_initialized:
            raise RuntimeError("Networks not initialized.")

        torch.save(self._best_conditional_flow.state_dict(), node_dir / "conditional_flow.pth")
        torch.save(self._best_marginal_flow.state_dict(), node_dir / "marginal_flow.pth")
        torch.save(self._init_parameters, node_dir / "init_parameters.pth")
        torch.save(self._total_epochs_trained, node_dir / "total_epochs_trained.pth")

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
        debug(f"Loading: {node_dir}")
        init_parameters = torch.load(node_dir / "init_parameters.pth")
        self._initialize_networks(init_parameters[0], init_parameters[1])

        self._best_conditional_flow.load_state_dict(torch.load(node_dir / "conditional_flow.pth"))
        self._best_marginal_flow.load_state_dict(torch.load(node_dir / "marginal_flow.pth"))

        if (node_dir / "embedding.pth").exists() and self._best_embedding is not None:
            self._best_embedding.load_state_dict(torch.load(node_dir / "embedding.pth"))

        _tep = node_dir / "total_epochs_trained.pth"
        self._total_epochs_trained = torch.load(_tep) if _tep.exists() else 0

    # ==================== Private Helpers ====================

    def _embed(self, conditions: Dict, train: bool = True, use_best_fit: bool = False):
        embedding = (
            self._best_embedding if use_best_fit and self._best_embedding is not None
            else self._embedding
        )
        embedding.train() if train else embedding.eval()
        return embedding(conditions)

    def _update_best_weights(self, network_type: str) -> None:
        if network_type == "conditional":
            self._best_conditional_flow.load_state_dict(
                self._conditional_flow.state_dict()
            )
            self._best_embedding.load_state_dict(
                self._embedding.state_dict()
            )
        else:
            self._best_marginal_flow.load_state_dict(
                self._marginal_flow.state_dict()
            )

    def _compute_discard_mask(self, theta, theta_logprob, conditions):
        u = self.simulator_instance.inverse(theta)
        s = self._embed(conditions, train=False, use_best_fit=True)

        u = u.expand(len(theta), *u.shape[1:]) if u.shape[0] == 1 else u
        s = s.expand(len(theta), *s.shape[1:]) if s.shape[0] == 1 else s

        u = u.to(self.device)
        self._conditional_flow.eval()
        log_prob = self._conditional_flow.log_prob(u.unsqueeze(0), s).squeeze(0).cpu()
        log_ratio = log_prob - theta_logprob.cpu()
        return log_ratio < self.log_ratio_threshold
