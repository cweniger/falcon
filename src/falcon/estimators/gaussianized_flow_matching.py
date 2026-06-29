"""Gaussianized flow-matching posterior estimator.

Combines a learned Gaussian preconditioner with flow-matching residual models:

  * A conditional Gaussian ``N(mu(x), Sigma)`` and an unconditional Gaussian
    ``N(mu0, Sigma0)`` (both in the prior's standard-normal latent space, with the
    GaussianFullCov machinery: MLP mean, EMA full covariance, eigendecomposition,
    and prior-width clamping of the covariance).
  * For each, a conditional flow-matching velocity field that learns the *residual*
    non-Gaussian structure in the whitened space ``w = Sigma^{-1/2}(theta_lat - mu)``,
    which is approximately N(0, I) -- so the flow only has to model what the Gaussian
    misses (multimodality, skew).

Sampling / density feed the same importance-sampling machinery as ``Flow``, adapted to
the standard-normal latent space: the latent prior is N(0, I) (not uniform), so the
analytic log N(0,I) is added to the importance weights, and a prior-width truncation
backstops runaway-wide proposals.
"""

import copy
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from falcon.core.logger import log, debug
from falcon.priors.product import TransformedPrior
from falcon.estimators.stepwise_base import StepwiseEstimator
from falcon.estimators.gaussian_fullcov import _GaussianPosterior
from falcon.estimators.flow_matching import (
    VelocityField, EMA, fm_loss, euler_sample, cnf_logprob,
)
from falcon.embeddings import instantiate_embedding


class _ClampedGaussian(_GaussianPosterior):
    """Gaussian preconditioner whose covariance is clamped at the prior width.

    In the standard-normal latent space the prior variance is 1, so a posterior is
    never *wider* than the prior. Clamping the residual-covariance eigenvalues at 1
    (the ``_output_std.clamp(max=1.0)`` in the base handles the std layer) stabilises
    parameters that are hard to constrain in a unimodal way -- they gracefully revert
    to the prior instead of inflating.
    """

    def _update_eigendecomp(self) -> None:
        eigvals, eigvecs = torch.linalg.eigh(self._residual_cov)
        self._residual_eigvals = eigvals.clamp(min=self.min_var, max=1.0)
        self._residual_eigvecs = eigvecs


class _GaussianizedFlow(nn.Module):
    """One Gaussian-preconditioner + flow-matching pair.

    Exposes the same surface as ``FlowDensity`` so the importance sampler is reusable:
        training_loss(theta_lat, s) -> dict of scalar losses
        sample(n, s)                -> (n, B, param) latent samples
        log_prob(theta_lat, s)      -> (N, B) latent log-density

    All densities/samples are over ``theta_lat`` (the standard-normal latent space);
    the constant whitening Jacobian ``-1/2 sum log lambda`` is folded into log_prob.
    """

    def __init__(self, param_dim: int, cond_dim: int, *, hidden_dim: int, num_layers: int,
                 momentum: float, min_var: float, eig_update_freq: int,
                 flow_hidden: int, flow_layers: int, time_dim: int, ema_decay: float,
                 sample_steps: int, density_steps: int, divergence: str, n_probe: int,
                 eval_chunk: int):
        super().__init__()
        self.param_dim = param_dim
        self.sample_steps = sample_steps
        self.density_steps = density_steps
        self.divergence = divergence
        self.n_probe = n_probe
        self.eval_chunk = eval_chunk

        self.gaussian = _ClampedGaussian(
            param_dim, cond_dim, hidden_dim=hidden_dim, num_layers=num_layers,
            momentum=momentum, min_var=min_var, eig_update_freq=eig_update_freq,
        )
        self.velocity = VelocityField(param_dim, cond_dim, flow_hidden, flow_layers, time_dim)
        self.velocity_ema = EMA.clone(self.velocity)
        self._ema = EMA(ema_decay)

    # ---- EMA ----
    def ema_update(self) -> None:
        self._ema.update(self.velocity_ema, self.velocity)

    # ---- whitening (float64, using the Gaussian's mu/Sigma) ----
    def _whiten(self, theta_lat: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
        V = self.gaussian._residual_eigvecs
        lam = self.gaussian._residual_eigvals
        diff = theta_lat.to(V.dtype) - mu
        return (diff @ V) / lam.sqrt()

    def _unwhiten(self, w: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
        V = self.gaussian._residual_eigvecs
        lam = self.gaussian._residual_eigvals
        return mu + (lam.sqrt() * w.to(V.dtype)) @ V.T

    def _logdet_whiten(self) -> torch.Tensor:
        return -0.5 * torch.log(self.gaussian._residual_eigvals).sum()

    # ---- training ----
    def training_loss(self, theta_lat: torch.Tensor, s: torch.Tensor) -> Dict[str, torch.Tensor]:
        nll = self.gaussian.loss(theta_lat, s)                # trains mu, updates EMA Sigma
        mu = self.gaussian._forward_mean(s).detach()
        w = self._whiten(theta_lat, mu).detach().float()      # flow sees a fixed whitened target
        fm = fm_loss(self.velocity, w, s.float())
        return {"nll": nll, "fm": fm, "total": nll + fm}

    # ---- sampling: (n, B, param) ----  chunked over n*B to bound memory
    @torch.no_grad()
    def sample(self, n: int, s: torch.Tensor) -> torch.Tensor:
        B, C, P = s.shape[0], s.shape[1], self.param_dim
        mu = self.gaussian._forward_mean(s)                          # (B, P) f64
        s_flat = s[None].expand(n, B, C).reshape(n * B, C).float()
        mu_flat = mu[None].expand(n, B, P).reshape(n * B, P)
        outs = []
        for i in range(0, n * B, self.eval_chunk):
            sc = s_flat[i:i + self.eval_chunk]
            w = euler_sample(self.velocity_ema, sc, P, self.sample_steps)
            outs.append(self._unwhiten(w, mu_flat[i:i + self.eval_chunk]))
        return torch.cat(outs, 0).reshape(n, B, P)                   # (n, B, P) f64

    # ---- density: (N, B, param) -> (N, B) ----  chunked over N*B
    def log_prob(self, theta_lat: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        N, B, P = theta_lat.shape
        C = s.shape[1]
        mu = self.gaussian._forward_mean(s).detach()                # (B, P)
        mu_flat = mu[None].expand(N, B, P).reshape(N * B, P)
        w_all = self._whiten(theta_lat.reshape(N * B, P), mu_flat).float()
        s_flat = s[None].expand(N, B, C).reshape(N * B, C).float()
        logdet = self._logdet_whiten()
        outs = []
        for i in range(0, N * B, self.eval_chunk):
            lp = cnf_logprob(self.velocity_ema, w_all[i:i + self.eval_chunk],
                             s_flat[i:i + self.eval_chunk],
                             self.density_steps, self.divergence, self.n_probe)
            outs.append(lp.to(mu.dtype) + logdet)
        return torch.cat(outs, 0).reshape(N, B)


class GaussianizedFlowMatching(StepwiseEstimator):
    """Gaussian-preconditioned flow-matching posterior estimator.

    Args:
        max_epochs, lr, gamma, embedding, device: as in other estimators.
        batch_size, early_stop_patience, prior_epochs, cache_*: training loop.
        hidden_dim, num_layers, momentum, min_var, eig_update_freq: Gaussian preconditioner.
        flow_hidden, flow_layers, time_dim, ema_decay: flow-matching velocity net.
        sample_steps: Euler steps for sampling.
        density_steps: Euler steps for the backward CNF density.
        divergence: "hutch" (Hutchinson, dimension-independent) or "exact" (d VJPs).
        n_probe: Hutchinson probe count (tunable; trades density noise vs cost).
        betas, lr_decay_factor, lr_patience: optimizer / scheduler.
        discard_samples, log_ratio_threshold: training-sample pruning.
        use_best_models: use best-checkpoint networks for sampling.
        num_proposals, proposal_mixture_beta: importance sampling.
        prior_sigma_bound, out_of_bounds_penalty: prior-width truncation backstop.
        nan_replacement: fallback for NaN/-inf log-weights.
    """

    def __init__(
        self,
        *,
        max_epochs: int = 100,
        lr: float = 1e-2,
        gamma: float = 0.5,
        embedding=None,
        device: Optional[str] = None,
        # training loop
        batch_size: int = 128,
        early_stop_patience: int = 16,
        prior_epochs: int = 0,
        cache_on_device: bool = False,
        cache_sync_every: int = 0,
        max_cache_samples: int = 0,
        # Gaussian preconditioner
        hidden_dim: int = 128,
        num_layers: int = 3,
        momentum: float = 0.01,
        min_var: float = 1e-20,
        eig_update_freq: int = 1,
        # flow-matching net
        flow_hidden: int = 256,
        flow_layers: int = 4,
        time_dim: int = 64,
        ema_decay: float = 0.999,
        sample_steps: int = 64,
        density_steps: int = 64,
        divergence: str = "hutch",
        n_probe: int = 4,
        eval_chunk: int = 50000,
        # optimizer
        betas: tuple = (0.9, 0.9),
        lr_decay_factor: float = 0.5,
        lr_patience: int = 8,
        # inference
        discard_samples: bool = False,
        log_ratio_threshold: float = -20.0,
        use_best_models: bool = True,
        num_proposals: int = 256,
        proposal_mixture_beta: float = 0.5,
        prior_sigma_bound: float = 6.0,
        out_of_bounds_penalty: float = 100.0,
        nan_replacement: float = -100.0,
    ):
        self.max_epochs = max_epochs
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
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.momentum = momentum
        self.min_var = min_var
        self.eig_update_freq = eig_update_freq
        self.flow_hidden = flow_hidden
        self.flow_layers = flow_layers
        self.time_dim = time_dim
        self.ema_decay = ema_decay
        self.sample_steps = sample_steps
        self.density_steps = density_steps
        self.divergence = divergence
        self.n_probe = n_probe
        self.eval_chunk = eval_chunk
        self.betas = betas
        self.lr_decay_factor = lr_decay_factor
        self.lr_patience = lr_patience
        self.discard_samples = discard_samples
        self.log_ratio_threshold = log_ratio_threshold
        self.use_best_models = use_best_models
        self.num_proposals = num_proposals
        self.proposal_mixture_beta = proposal_mixture_beta
        self.prior_sigma_bound = prior_sigma_bound
        self.out_of_bounds_penalty = out_of_bounds_penalty
        self.nan_replacement = nan_replacement

    # ==================== Setup ====================

    def setup(self, simulator_instance, theta_key=None, condition_keys=None):
        if not isinstance(simulator_instance, TransformedPrior):
            raise TypeError(
                f"GaussianizedFlowMatching requires a TransformedPrior (e.g. Product), "
                f"got {type(simulator_instance).__name__}."
            )
        super().setup(simulator_instance, theta_key, condition_keys)

        if self.device:
            self.device = torch.device(self.device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            debug(f"Auto-detected device: {self.device}")

        self._embedding = instantiate_embedding(self.embedding).to(self.device)

        self._cond = None
        self._marg = None
        self._best_cond = None
        self._best_marg = None
        self._best_embedding = None
        self._init_parameters = None
        self._best_loss = float("inf")
        self._optimizer = None
        self._scheduler = None

    # ==================== Initialization ====================

    def _build_module(self, param_dim: int, cond_dim: int) -> _GaussianizedFlow:
        return _GaussianizedFlow(
            param_dim, cond_dim,
            hidden_dim=self.hidden_dim, num_layers=self.num_layers,
            momentum=self.momentum, min_var=self.min_var, eig_update_freq=self.eig_update_freq,
            flow_hidden=self.flow_hidden, flow_layers=self.flow_layers, time_dim=self.time_dim,
            ema_decay=self.ema_decay, sample_steps=self.sample_steps,
            density_steps=self.density_steps, divergence=self.divergence, n_probe=self.n_probe,
            eval_chunk=self.eval_chunk,
        ).to(self.device)

    def _initialize_networks(self, theta: torch.Tensor, conditions: Dict) -> None:
        debug("Initializing GaussianizedFlowMatching networks...")
        self._init_parameters = [theta, conditions]

        conditions_device = {k: v.to(self.device) for k, v in conditions.items()}
        s = self._embed(conditions_device, train=False).detach()
        theta_lat = self.simulator_instance.inverse(theta.to(self.device), mode="standard_normal")

        param_dim, cond_dim = theta_lat.shape[1], s.shape[1]
        self._cond = self._build_module(param_dim, cond_dim)
        self._marg = self._build_module(param_dim, cond_dim)
        self._best_cond = copy.deepcopy(self._cond)
        self._best_marg = copy.deepcopy(self._marg)
        self._best_embedding = copy.deepcopy(self._embedding)

        params = [
            p for m in (self._embedding, self._cond, self._marg)
            for p in m.parameters() if p.requires_grad
        ]
        self._optimizer = AdamW(params, lr=self.lr, betas=self.betas)
        self._scheduler = ReduceLROnPlateau(
            self._optimizer, mode="min", factor=self.lr_decay_factor, patience=self.lr_patience,
        )
        self.networks_initialized = True
        debug(f"Networks initialized: param_dim={param_dim}, cond_dim={cond_dim}")

    def _embed(self, conditions: Dict, train: bool = True, use_best_fit: bool = False):
        embedding = (
            self._best_embedding if use_best_fit and self._best_embedding is not None
            else self._embedding
        )
        embedding.train() if train else embedding.eval()
        return embedding(conditions)

    # ==================== Train / Val ====================

    def _unpack(self, batch, phase: str):
        theta = self._to_tensor(batch[f"{self.theta_key}.value"]).to(self.device)
        theta_logprob = self._to_tensor(batch[f"{self.theta_key}.log_prob"])
        conditions = {
            k: self._to_tensor(batch[f"{k}.value"], self.device)
            for k in self.condition_keys if f"{k}.value" in batch
        }
        ts = time.time()
        self.history[f"{phase}_ids"].extend((ts, i) for i in batch._ids.tolist())
        theta_lat = self.simulator_instance.inverse(theta, mode="standard_normal")
        return theta, theta_logprob, conditions, theta_lat

    def _modules_loss(self, theta_lat, s, train: bool):
        mode = (self._cond.train, self._marg.train) if train else (self._cond.eval, self._marg.eval)
        mode[0](); mode[1]()
        cond = self._cond.training_loss(theta_lat, s)
        marg = self._marg.training_loss(theta_lat, s * 0)
        return cond, marg

    def train_step(self, batch) -> Dict[str, float]:
        theta, theta_logprob, conditions, theta_lat = self._unpack(batch, "train")
        if not self.networks_initialized:
            self._initialize_networks(theta, conditions)

        s = self._embed(conditions, train=True)
        self._optimizer.zero_grad()
        cond, marg = self._modules_loss(theta_lat, s, train=True)
        (cond["total"] + marg["total"]).backward()
        self._optimizer.step()
        self._cond.ema_update()
        self._marg.ema_update()

        if self.discard_samples:
            batch.discard(self._compute_discard_mask(theta_lat, theta_logprob, s.detach()))

        return {"loss": cond["fm"].item(), "loss_aux": marg["fm"].item(),
                "nll": cond["nll"].item()}

    def val_step(self, batch) -> Dict[str, float]:
        _, _, conditions, theta_lat = self._unpack(batch, "val")
        s = self._embed(conditions, train=False)
        with torch.no_grad():
            cond, marg = self._modules_loss(theta_lat, s, train=False)
        return {"loss": cond["fm"].item(), "loss_aux": marg["fm"].item()}

    def on_epoch_end(self, epoch: int, val_metrics: Dict[str, float]) -> Optional[Dict[str, float]]:
        val_loss = val_metrics.get("loss", float("inf")) + val_metrics.get("loss_aux", 0.0)
        if val_loss < self._best_loss:
            self._best_loss = val_loss
            self._best_cond.load_state_dict(self._cond.state_dict())
            self._best_marg.load_state_dict(self._marg.state_dict())
            self._best_embedding.load_state_dict(self._embedding.state_dict())
            log({"checkpoint": epoch})

        self._scheduler.step(val_metrics.get("loss", float("inf")))
        lr = self._optimizer.param_groups[0]["lr"]
        log({"lr": lr})
        return {"lr": lr}

    def _compute_discard_mask(self, theta_lat, theta_logprob, s):
        self._cond.eval()
        with torch.no_grad():
            log_prob = self._cond.log_prob(theta_lat.unsqueeze(0), s).squeeze(0).cpu()
        return (log_prob - theta_logprob.cpu()) < self.log_ratio_threshold

    # ==================== Sampling ====================

    def sample_prior(self, num_samples: int, conditions=None) -> dict:
        if conditions:
            raise ValueError("Conditions are not supported for sample_prior.")
        samples = self.simulator_instance.simulate_batch(num_samples)
        return {"value": samples, "log_prob": np.zeros(num_samples)}

    def sample_posterior(self, num_samples: int, conditions=None) -> dict:
        if not self.networks_initialized:
            return self.sample_prior(num_samples)
        samples, logprob = self._importance_sample(num_samples, "posterior", conditions or {})
        return {"value": samples.numpy(), "log_prob": logprob.numpy()}

    def sample_proposal(self, num_samples: int, conditions=None) -> dict:
        if self._total_epochs_trained < self.prior_epochs or not self.networks_initialized:
            return self.sample_prior(num_samples)
        samples, logprob = self._importance_sample(num_samples, "proposal", conditions or {})
        log({"sample_proposal:mean": samples.mean().item(),
             "sample_proposal:std": samples.std().item(),
             "sample_proposal:logprob": logprob.mean().item()})
        return {"value": samples.numpy(), "log_prob": logprob.numpy()}

    def _importance_sample(self, num_samples: int, mode: str, conditions: Dict):
        assert conditions, "Conditions must be provided."
        conditions = {k: self._to_tensor(v, self.device) for k, v in conditions.items()}

        use_best = self.use_best_models and self._best_cond is not None
        cond_net = self._best_cond if use_best else self._cond
        marg_net = self._best_marg if use_best else self._marg
        s = self._embed(conditions, train=False, use_best_fit=use_best).detach()
        s = s.expand(num_samples, *s.shape[1:])

        cond_net.eval(); marg_net.eval()

        # Multiple-importance-sampling proposal mixture (balance heuristic), as in Flow.
        n_cond = max(0, min(self.num_proposals, int(round(self.proposal_mixture_beta * self.num_proposals))))
        n_marg = self.num_proposals - n_cond
        parts = []
        if n_cond > 0:
            parts.append(cond_net.sample(n_cond, s))
        if n_marg > 0:
            parts.append(marg_net.sample(n_marg, s * 0))
        proposals = torch.cat(parts, dim=0)                       # (num_proposals, num_samples, P)

        log_prob_cond = cond_net.log_prob(proposals, s)
        log_prob_marg = marg_net.log_prob(proposals, s * 0)

        # Latent prior is N(0, I) (NOT uniform): it enters the weights analytically.
        log_prior = -0.5 * (proposals.pow(2).sum(-1) + self.param_dim * np.log(2 * np.pi))

        # Prior-width truncation backstop (analog of Flow's hypercube clipping).
        oob = (proposals.abs() > self.prior_sigma_bound).any(dim=-1).to(log_prior.dtype)
        mask = oob * self.out_of_bounds_penalty

        # Balance-heuristic mixture density sum_k (n_k / N) g_k.
        total = n_cond + n_marg
        mix = []
        if n_cond > 0:
            mix.append(np.log(n_cond / total) + log_prob_cond)
        if n_marg > 0:
            mix.append(np.log(n_marg / total) + log_prob_marg)
        log_g_mix = torch.logaddexp(mix[0], mix[1]) if len(mix) == 2 else mix[0]

        if mode == "proposal":
            log_target = self.gamma / (1.0 + self.gamma) * log_prob_cond + log_prior
        else:  # posterior: c/m * prior
            log_target = log_prob_cond - log_prob_marg + log_prior

        log_weights = log_target - log_g_mix - mask
        log_weights = torch.nan_to_num(log_weights, nan=self.nan_replacement, neginf=self.nan_replacement)
        log_weights = log_weights - torch.logsumexp(log_weights, dim=0, keepdim=True)
        weights = torch.exp(log_weights)

        n_eff = 1.0 / (weights ** 2).sum(dim=0).cpu().numpy()
        log({"importance_sample:n_eff_min": float(n_eff.min()),
             "importance_sample:n_eff_max": float(n_eff.max())})

        idx = torch.multinomial(weights.T, 1, replacement=True).squeeze(-1)
        samples_lat = proposals[idx, torch.arange(num_samples), :]
        samples = self.simulator_instance.forward(samples_lat, mode="standard_normal").cpu()
        logprob = log_prob_cond[idx, torch.arange(num_samples)].cpu()
        return samples, logprob.detach()

    # ==================== Save / Load ====================

    def save(self, node_dir: Path) -> None:
        node_dir = Path(node_dir)
        if not self.networks_initialized:
            raise RuntimeError("Networks not initialized.")
        torch.save(self._best_cond.state_dict(), node_dir / "conditional.pth")
        torch.save(self._best_marg.state_dict(), node_dir / "marginal.pth")
        torch.save(self._best_embedding.state_dict(), node_dir / "embedding.pth")
        torch.save(self._init_parameters, node_dir / "init_parameters.pth")
        torch.save(self._total_epochs_trained, node_dir / "total_epochs_trained.pth")

        torch.save(self.history["train_ids"], node_dir / "train_id_history.pth")
        torch.save(self.history["val_ids"], node_dir / "validation_id_history.pth")
        torch.save(self.history["epochs"], node_dir / "epochs.pth")
        torch.save(self.history["train_loss"], node_dir / "loss_train_posterior.pth")
        torch.save(self.history["val_loss"], node_dir / "loss_val_posterior.pth")
        torch.save(self.history["n_samples"], node_dir / "n_samples_total.pth")
        torch.save(self.history["elapsed_min"], node_dir / "elapsed_minutes.pth")

    def load(self, node_dir: Path) -> None:
        node_dir = Path(node_dir)
        init = torch.load(node_dir / "init_parameters.pth")
        self._initialize_networks(init[0], init[1])

        self._best_cond.load_state_dict(torch.load(node_dir / "conditional.pth"))
        self._best_marg.load_state_dict(torch.load(node_dir / "marginal.pth"))
        self._cond.load_state_dict(self._best_cond.state_dict())
        self._marg.load_state_dict(self._best_marg.state_dict())
        if (node_dir / "embedding.pth").exists():
            self._best_embedding.load_state_dict(torch.load(node_dir / "embedding.pth"))
            self._embedding.load_state_dict(self._best_embedding.state_dict())

        tep = node_dir / "total_epochs_trained.pth"
        self._total_epochs_trained = torch.load(tep) if tep.exists() else 0
