"""Flow-based posterior estimation (was SNPE_A)."""

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from falcon.core.logger import log, debug
from falcon.estimators.flow_density import FlowDensity
from falcon.estimators.torch_model import NetworkGroup, TorchModel, to_numpy_tree
from falcon.embeddings import instantiate_embedding


class Flow(TorchModel):
    """Flow-based posterior estimation using a conditional + marginal flow pair.

    Training runs in rounds of epochs on fixed data; see ``RoundTrainer``.
    The conditional flow (with the embedding) and the marginal flow are
    promoted to best networks independently; a round is accepted when the
    conditional flow improves.

    Args:
        max_epochs: Maximum epochs per round.
        net_type: Flow architecture (``zuko_nice``, ``nsf``, ``maf``, ``zuko_gf``, ...).
        lr: Learning rate; reset at the start of every round.
        gamma: Proposal tempering coefficient.
        embedding: Embedding config dict (with ``_target_`` etc.) or ``None``.
        device: Device string (e.g. ``"cuda:0"``); auto-detected if ``None``.
        batch_size: Mini-batch size.
        patience_epochs: End the round once the best validation loss is this
            many epochs old (checked at validations).
        val_every_epochs: Validate after every N-th epoch (and at the last epoch).
        max_rounds: Maximum number of rounds (``None`` = unlimited).
        patience_rounds: Stop training after this many rejected rounds in a row.
        prior_rounds: Rounds that simulate from the prior before switching to
            the learned proposal. The first round always uses the prior.
        cache_on_device: Cache training data on the estimator device.
        max_cache_samples: Cap on cached training samples (0 = all).
        theta_norm: Normalise parameter space online.
        norm_momentum: EMA momentum for online normalisation.
        adaptive_momentum: Adaptive momentum for normalisation.
        use_log_update: Use log-space normalisation update.
        betas: AdamW beta coefficients.
        lr_decay_factor: LR decay factor for plateau scheduler (1.0 = no decay).
        lr_patience_epochs: Epochs without validation improvement before LR decay.
        discard_samples: Run a discard sweep after every accepted round.
        log_ratio_threshold: Log-ratio cutoff for discarding.
        sample_reference_posterior: Sample reference posterior for proposals.
        num_proposals: Importance sampling proposal count.
        proposal_mixture_beta: Fraction of proposals drawn from the conditional
            flow in the multiple-importance-sampling mixture (balance heuristic);
            the rest are drawn from the marginal flow. ``1.0`` = conditional
            only (reproduces the single-proposal behaviour), ``0.0`` = marginal
            only, ``0.5`` = even defensive mixture.
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
        patience_epochs: int = 16,
        val_every_epochs: int = 1,
        max_rounds: Optional[int] = None,
        patience_rounds: int = 10,
        prior_rounds: int = 0,
        cache_on_device: bool = False,
        max_cache_samples: int = 0,
        # Network
        theta_norm: bool = True,
        norm_momentum: float = 1e-2,
        adaptive_momentum: bool = False,
        use_log_update: bool = False,
        # Optimizer
        betas: tuple = (0.9, 0.9),
        lr_decay_factor: float = 1.0,
        lr_patience_epochs: int = 8,
        # Inference
        discard_samples: bool = True,
        log_ratio_threshold: float = -20.0,
        sample_reference_posterior: bool = False,
        num_proposals: int = 256,
        proposal_mixture_beta: float = 0.5,
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
        self.patience_epochs = patience_epochs
        self.val_every_epochs = val_every_epochs
        self.max_rounds = max_rounds
        self.patience_rounds = patience_rounds
        self.prior_rounds = prior_rounds
        self.cache_on_device = cache_on_device
        self.max_cache_samples = max_cache_samples
        self.theta_norm = theta_norm
        self.norm_momentum = norm_momentum
        self.adaptive_momentum = adaptive_momentum
        self.use_log_update = use_log_update
        self.betas = betas
        self.lr_decay_factor = lr_decay_factor
        self.lr_patience_epochs = lr_patience_epochs
        self.discard_samples = discard_samples
        self.log_ratio_threshold = log_ratio_threshold
        self.sample_reference_posterior = sample_reference_posterior
        self.num_proposals = num_proposals
        self.proposal_mixture_beta = proposal_mixture_beta
        self.reference_samples = reference_samples
        self.hypercube_bound = hypercube_bound
        self.out_of_bounds_penalty = out_of_bounds_penalty
        self.nan_replacement = nan_replacement

    def setup(self, simulator_instance, theta_key=None, condition_keys=None):
        super().setup(simulator_instance, theta_key, condition_keys)
        self.param_dim = simulator_instance.param_dim

        self._embedding = None
        self._conditional_flow = None
        self._marginal_flow = None
        self._optimizer = None
        self._scheduler = None

        self._history = {"theta_mins": [], "theta_maxs": []}

    @property
    def history(self):
        return self._history

    # ==================== Networks ====================

    def init_from_batch(self, batch):
        theta = self._to_tensor(batch[f"{self.theta_key}.value"])
        conditions = {
            k: self._to_array(batch[f"{k}.value"])
            for k in self.condition_keys if f"{k}.value" in batch
        }
        u = self.simulator_instance.inverse(theta)
        return {"theta": self._to_array(u), "conditions": conditions}

    def build(self, init_tree) -> None:
        debug("Initializing networks...")
        theta = self._to_tensor(init_tree["theta"], self.device)
        conditions = {k: self._to_tensor(v, self.device) for k, v in init_tree["conditions"].items()}

        self._embedding = instantiate_embedding(self.embedding).to(self.device)
        self._embedding.eval()
        with torch.no_grad():
            s = self._embedding(conditions).detach()

        self._conditional_flow = self._create_flow(theta, s, is_conditional=True).to(self.device)
        self._marginal_flow = self._create_flow(theta, s, is_conditional=False).to(self.device)
        self._set_groups({
            "conditional": NetworkGroup(
                {"flow": self._conditional_flow, "embedding": self._embedding}, "loss",
                embedding=self._embedding,
            ),
            "marginal": NetworkGroup({"flow": self._marginal_flow}, "loss_aux"),
        })
        debug("Networks initialized.")

    def _build_optimizer(self) -> None:
        parameters = (
            list(self._conditional_flow.parameters())
            + list(self._marginal_flow.parameters())
            + list(self._embedding.parameters())
        )
        self._optimizer = AdamW(parameters, lr=self.lr, betas=self.betas)
        self._build_scheduler()

    def _build_scheduler(self) -> None:
        # Stepped once per validation, so the patience is converted from epochs
        self._scheduler = (
            ReduceLROnPlateau(
                self._optimizer,
                mode="min",
                factor=self.lr_decay_factor,
                patience=-(-self.lr_patience_epochs // self.val_every_epochs),
            )
            if self.lr_decay_factor < 1.0 else None
        )

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
        theta = self._to_tensor(batch[f"{self.theta_key}.value"])
        theta_logprob = self._to_tensor(batch[f"{self.theta_key}.log_prob"])
        conditions = {
            k: self._to_tensor(batch[f"{k}.value"], self.device)
            for k in self.condition_keys if f"{k}.value" in batch
        }

        log({f"{phase}:theta_logprob_min": theta_logprob.min().item()})
        log({f"{phase}:theta_logprob_max": theta_logprob.max().item()})

        u = self.simulator_instance.inverse(theta).to(self.device)
        return theta, theta_logprob, u, conditions

    def _compute_flow_losses(self, u, s, train: bool):
        self._conditional_flow.train(train)
        self._marginal_flow.train(train)

        loss_cond = self._conditional_flow.loss(u, s).mean()
        s_marginal = s.detach() * 0 if train else s * 0
        loss_marg = self._marginal_flow.loss(u, s_marginal).mean()

        return loss_cond, loss_marg

    def train_step(self, batch) -> Dict[str, float]:
        theta, _, u, conditions = self._unpack_batch(batch, "train")
        if self._optimizer is None:
            self._build_optimizer()

        s = self._summary("conditional", conditions, train=True)

        with torch.no_grad():
            self._history["theta_mins"].append(theta.min(dim=0).values.cpu().numpy())
            self._history["theta_maxs"].append(theta.max(dim=0).values.cpu().numpy())

        self._optimizer.zero_grad()
        loss_cond, loss_marg = self._compute_flow_losses(u, s, train=True)
        (loss_cond + loss_marg).backward()
        self._optimizer.step()

        return {"loss": loss_cond.item(), "loss_aux": loss_marg.item()}

    def val_step(self, batch) -> Dict[str, float]:
        _, _, u, conditions = self._unpack_batch(batch, "val")
        s = self._summary("conditional", conditions)
        loss_cond, loss_marg = self._compute_flow_losses(u, s, train=False)
        return {"loss": loss_cond.item(), "loss_aux": loss_marg.item()}

    def on_validation_end(self, epoch: int, val_metrics: Dict[str, float]) -> Optional[Dict[str, float]]:
        if self._scheduler is not None:
            self._scheduler.step(val_metrics.get("loss", float("inf")))
        lr = self._optimizer.param_groups[0]["lr"]
        log({"lr": lr})

        return {"lr": lr}

    def on_round_start(self) -> None:
        if self._optimizer is None:
            return
        for group in self._optimizer.param_groups:
            group["lr"] = self.lr
        self._build_scheduler()

    def discard_test(self, batch):
        theta = self._to_tensor(batch[f"{self.theta_key}.value"])
        theta_logprob = self._to_tensor(batch[f"{self.theta_key}.log_prob"])
        conditions = {
            k: self._to_tensor(batch[f"{k}.value"], self.device)
            for k in self.condition_keys if f"{k}.value" in batch
        }
        u = self.simulator_instance.inverse(theta).to(self.device)
        s = self._summary("conditional", conditions)

        self._conditional_flow.eval()
        log_prob = self._conditional_flow.log_prob(u.unsqueeze(0), s).squeeze(0).cpu()
        log_ratio = log_prob - theta_logprob.cpu()
        return log_ratio < self.log_ratio_threshold

    # ==================== Sampling ====================

    def _sample_prior(self, num_samples: int) -> dict:
        samples = self.simulator_instance.simulate_batch(num_samples)
        log_prob = np.ones(num_samples) * (-np.log(2 * self.hypercube_bound) ** self.param_dim)
        return {'value': samples, 'log_prob': log_prob}

    def _sample(self, num_samples: int, conditions, mode: str) -> dict:
        if mode == "proposal" and self.sample_reference_posterior:
            post_samples, _ = self._importance_sample(
                self.reference_samples, mode="posterior", conditions=conditions
            )
            mean, std = post_samples.mean(dim=0).cpu(), post_samples.std(dim=0).cpu()
            log({f"sample_proposal:posterior_mean_{i}": mean[i].item() for i in range(len(mean))})
            log({f"sample_proposal:posterior_std_{i}": std[i].item() for i in range(len(std))})

        samples, logprob = self._importance_sample(num_samples, mode=mode, conditions=conditions)
        if mode == "proposal":
            log({
                "sample_proposal:mean": samples.mean().item(),
                "sample_proposal:std": samples.std().item(),
                "sample_proposal:logprob": logprob.mean().item(),
            })
        return {'value': samples.numpy(), 'log_prob': logprob.numpy()}

    def _importance_sample(self, num_samples: int, mode: str = "posterior", conditions: Dict = {}):
        assert conditions, "Conditions must be provided."
        conditions = {k: self._to_tensor(v, self.device) for k, v in conditions.items()}

        # Sampling runs in the sample actor, whose networks are the best networks
        conditional_net = self._conditional_flow
        marginal_net = self._marginal_flow
        self._embedding.eval()
        s = self._embedding(conditions)

        s = s.expand(num_samples, *s.shape[1:])

        conditional_net.eval()
        marginal_net.eval()

        # Multiple importance sampling: draw the proposals from a mixture of the
        # conditional and marginal flows -- beta of them from the conditional and
        # (1 - beta) from the marginal -- keeping the total proposal count fixed.
        # The marginal flow is the broad/defensive component that covers the tails
        # the (tempered) conditional under-covers; the conditional gives efficiency
        # near the mode. The mixture-density weighting below is the balance
        # heuristic, which is also a defensive mixture in Hesterberg's sense.
        # Refs:
        #   Hesterberg (1995), "Weighted Average Importance Sampling and
        #     Defensive Mixture Distributions", Technometrics 37(2):185-194.
        #     DOI: 10.1080/00401706.1995.10484303
        #   Veach & Guibas (1995), "Optimally Combining Sampling Techniques
        #     for Monte Carlo Rendering", SIGGRAPH '95, pp. 419-428
        #     DOI: 10.1145/218380.218498
        #     Introduces multiple importance sampling; includes the balance heuristic.
        n_cond = int(round(self.proposal_mixture_beta * self.num_proposals))
        n_cond = max(0, min(self.num_proposals, n_cond))
        n_marg = self.num_proposals - n_cond

        parts = []
        if n_cond > 0:
            parts.append(conditional_net.sample(n_cond, s).detach())
        if n_marg > 0:
            parts.append(marginal_net.sample(n_marg, s * 0).detach())
        samples_proposals = torch.cat(parts, dim=0)

        log({
            "importance_sample:proposal_mean": samples_proposals.mean().item(),
            "importance_sample:proposal_std": samples_proposals.std().item(),
        })

        # Both densities must be evaluated on the full pooled set, regardless of
        # which flow drew each sample (this is what the balance heuristic needs).
        log_prob_cond = conditional_net.log_prob(samples_proposals, s)
        log_prob_marg = marginal_net.log_prob(samples_proposals, s * 0)

        mask = (samples_proposals < -self.hypercube_bound) | (samples_proposals > self.hypercube_bound)
        mask = mask.any(dim=-1).float() * self.out_of_bounds_penalty

        # Balance-heuristic mixture density: sum_k (n_k / N) * g_k(theta).
        total = n_cond + n_marg
        mix_terms = []
        if n_cond > 0:
            mix_terms.append(np.log(n_cond / total) + log_prob_cond)
        if n_marg > 0:
            mix_terms.append(np.log(n_marg / total) + log_prob_marg)
        log_g_mix = (
            torch.logaddexp(mix_terms[0], mix_terms[1])
            if len(mix_terms) == 2 else mix_terms[0]
        )

        # Unnormalised target: tempered posterior c^{gamma/(1+gamma)} for the
        # proposal distribution; importance-corrected posterior c/m otherwise.
        if mode == "proposal":
            log_target = self.gamma / (1.0 + self.gamma) * log_prob_cond
        else:
            log_target = log_prob_cond - log_prob_marg

        log_weights = log_target - log_g_mix - mask

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

    # ==================== Legacy checkpoints ====================

    def load_legacy(self, node_dir: Path):
        """Checkpoint files written before ``best_state.npz`` existed."""
        if not (node_dir / "conditional_flow.pth").exists():
            return None
        load = self._legacy_load
        theta, conditions = load(node_dir / "init_parameters.pth")
        conditional = {"flow": to_numpy_tree(load(node_dir / "conditional_flow.pth"))}
        if (node_dir / "embedding.pth").exists():
            conditional["embedding"] = to_numpy_tree(load(node_dir / "embedding.pth"))
        else:  # the embedding has no state
            conditional["embedding"] = {}
        return {
            "meta": self._legacy_meta(node_dir),
            "init": {"theta": self._to_array(theta), "conditions": to_numpy_tree(conditions)},
            "groups": {
                "conditional": conditional,
                "marginal": {"flow": to_numpy_tree(load(node_dir / "marginal_flow.pth"))},
            },
        }
