"""Flow-matching posterior estimation with a truncated-prior proposal.

Adopted from the LDC MBHB reference loop (``run_trunc7b.py``). A conditional
flow ``q_c(u | x)`` and a marginal flow ``q_m(u)`` over the standard-normal
latent ``u`` of the prior are trained by flow matching: a velocity field
``v(w, t, s)`` regresses the straight-line bridge from N(0, I) at ``t = 0`` to
the data at ``t = 1``. Each flow works in the frame ``w`` of an exact ZCA
whitener that is refit from the round's training data, so the velocity field
sees a unit-scale target however far the buffer has contracted. Samples come
from integrating the ODE forward with Euler steps, densities from the
continuous change of variables with the ODE run backward.

Proposals are the prior truncated to a region read off ``q_c`` and moved once
per round by the region ladder (``falcon.estimators.region_ladder``). Since
the training data is the truncated prior, ``q_c`` itself estimates the
posterior: posterior samples are ``q_c`` draws cut at the ``x_sigma`` contour,
without importance weights.
"""

import copy
import hashlib
import math
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from falcon.core.logger import debug, info, log, warning
from falcon.embeddings import instantiate_embedding
from falcon.estimators.region_ladder import FlowPair, LadderConfig, RegionLadder, leak, log_normal
from falcon.estimators.torch_model import NetworkGroup, TorchModel, load_module_state, module_state
from falcon.priors.product import TransformedPrior


# ==================== Velocity field ====================


class GaussianFourierTime(nn.Module):
    """Random Fourier features of the scalar time."""

    def __init__(self, dim: int, scale: float = 3.0):
        super().__init__()
        if dim % 2:
            raise ValueError(f"time_dim must be even, got {dim}")
        self.register_buffer("freqs", torch.randn(dim // 2) * scale)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        proj = 2 * math.pi * t * self.freqs
        return torch.cat([proj.sin(), proj.cos()], dim=-1)


class VelocityField(nn.Module):
    """MLP velocity field ``v(w, t, s)`` of whitened parameters, time and conditioning."""

    def __init__(self, param_dim: int, cond_dim: int, hidden: int = 512, layers: int = 6,
                 time_dim: int = 32, layernorm: bool = False):
        super().__init__()
        self.param_dim = param_dim
        self.time_embed = GaussianFourierTime(time_dim)
        dims = [param_dim + time_dim + cond_dim] + [hidden] * layers
        net = []
        for d_in, d_out in zip(dims[:-1], dims[1:]):
            net.append(nn.Linear(d_in, d_out))
            if layernorm:
                net.append(nn.LayerNorm(d_out))
            net.append(nn.SiLU())
        net.append(nn.Linear(dims[-1], param_dim))
        self.net = nn.Sequential(*net)

    def forward(self, w: torch.Tensor, t: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        if t.ndim == 1:
            t = t[:, None]
        return self.net(torch.cat([w, self.time_embed(t), s], dim=-1))


@torch.no_grad()
def copy_buffers(dst: nn.Module, src: nn.Module, skip_float: bool = False) -> None:
    """Copy the buffers of ``src`` into ``dst`` by name, creating those ``dst`` lacks (lazy buffers).

    With ``skip_float``, floating-point buffers that ``dst`` already has in the
    same shape and dtype are left alone.
    """
    for name, buffer in src.named_buffers():
        path, _, key = name.rpartition(".")
        module = dst.get_submodule(path) if path else dst
        current = module._buffers.get(key)
        if current is None or current.shape != buffer.shape or current.dtype != buffer.dtype:
            module._buffers[key] = buffer.detach().clone()
        elif not (skip_float and current.is_floating_point()):
            current.copy_(buffer)


@torch.no_grad()
def ema_update(ema: nn.Module, model: nn.Module, decay: float) -> None:
    """Move the parameters and floating-point buffers of ``ema`` toward ``model``.

    Buffers such as the running statistics of an embedding's normalisation
    are averaged like the parameters: copied, they would jump to the latest
    statistics while the averaged weights still expect the old ones. Other
    buffers, and buffers ``ema`` does not have yet (lazy buffers), are copied.
    """
    params = dict(model.named_parameters())
    for name, p in ema.named_parameters():
        p.mul_(decay).add_(params[name].detach(), alpha=1.0 - decay)
    current = dict(ema.named_buffers())
    for name, buffer in model.named_buffers():
        mine = current.get(name)
        if (mine is not None and mine.is_floating_point() and mine.shape == buffer.shape
                and mine.dtype == buffer.dtype):
            mine.mul_(decay).add_(buffer.detach(), alpha=1.0 - decay)
    copy_buffers(ema, model, skip_float=True)


# ==================== Flow matching ====================


def fm_loss_late(net: VelocityField, w1: torch.Tensor, s: torch.Tensor,
                 k: float = 8.0, mix: float = 0.5) -> torch.Tensor:
    """Flow-matching loss with late-time-weighted time sampling.

    ``t`` is drawn from ``p(t) = (1 - mix) + mix * k * t^(k-1)``, so the sharp
    end of the probability path gets more supervision. Antithetic in the bridge
    noise only; mirroring ``t -> 1 - t`` would cancel the tilt.
    """
    u = torch.rand(w1.shape[0], 1, device=w1.device, dtype=w1.dtype)
    late = torch.rand_like(u) < mix
    t = torch.where(late, u.pow(1.0 / k), u)
    w0 = torch.randn_like(w1)
    w0 = torch.cat([w0, -w0])
    t = torch.cat([t, t])
    w1 = w1.repeat(2, 1)
    s = s.repeat(2, 1)
    wt = (1 - t) * w0 + t * w1
    return (net(wt, t, s) - (w1 - w0)).pow(2).mean()


@torch.no_grad()
def val_fm_loss(net: VelocityField, w: torch.Tensor, s: torch.Tensor, n_rep: int = 4) -> float:
    """Flow-matching loss on a seeded time grid, so that successive validations are paired."""
    generator = torch.Generator(device="cpu").manual_seed(12345)
    total = 0.0
    for _ in range(n_rep):
        t = torch.rand(w.shape[0], 1, generator=generator).to(w.device, w.dtype)
        w0 = torch.randn_like(w)
        wt = (1 - t) * w0 + t * w
        total += float((net(wt, t, s) - (w - w0)).pow(2).mean())
    return total / n_rep


@torch.no_grad()
def euler_sample(net: VelocityField, s: torch.Tensor, steps: int) -> torch.Tensor:
    """One draw per row of ``s``: the ODE integrated forward from t = 0 to 1."""
    m = s.shape[0]
    w = torch.randn(m, net.param_dim, device=s.device, dtype=s.dtype)
    dt = 1.0 / steps
    for i in range(steps):
        t = torch.full((m, 1), i * dt, device=s.device, dtype=w.dtype)
        w = w + dt * net(w, t, s)
    return w


def _velocity_and_divergence(net, w, t, s, divergence: str, n_probe: int):
    """Velocity and its divergence: exact (one VJP per dimension) or Hutchinson (``n_probe`` probes)."""
    with torch.enable_grad():
        w = w.detach().requires_grad_(True)
        v = net(w, t, s)
        div = torch.zeros(w.shape[0], device=w.device, dtype=w.dtype)
        if divergence == "exact":
            for i in range(w.shape[1]):
                g = torch.autograd.grad(v[:, i].sum(), w, retain_graph=i < w.shape[1] - 1)[0]
                div = div + g[:, i]
        else:
            for _ in range(n_probe):
                eps = torch.randint(0, 2, w.shape, device=w.device, dtype=w.dtype) * 2 - 1
                g = torch.autograd.grad(v, w, grad_outputs=eps, retain_graph=True)[0]
                div = div + (g * eps).sum(1)
            div = div / n_probe
    return v.detach(), div.detach()


def cnf_logprob(net: VelocityField, w: torch.Tensor, s: torch.Tensor, steps: int,
                divergence: str = "exact", n_probe: int = 4) -> torch.Tensor:
    """Log density at ``w``: ``log N(z(0); 0, I) + int_1^0 div v dt``, with the ODE run backward."""
    z = w
    logdet = torch.zeros(w.shape[0], device=w.device, dtype=w.dtype)
    dt = -1.0 / steps
    for i in range(steps):
        t = torch.full((z.shape[0], 1), 1.0 + i * dt, device=z.device, dtype=z.dtype)
        v, d = _velocity_and_divergence(net, z, t, s, divergence, n_probe)
        z = z + dt * v
        logdet = logdet + dt * d
    base = -0.5 * (z.pow(2).sum(1) + z.shape[1] * math.log(2 * math.pi))
    return base + logdet


# ==================== Whitened flow ====================


class Whitener(nn.Module):
    """Exact ZCA whitener of the latent parameters, refit once per round.

    Uses the symmetric root ``Sigma^{-1/2} = V diag(lambda^{-1/2}) V^T``, a
    function of ``Sigma`` alone: eigenvectors of near-degenerate directions may
    jitter and flip from one refit to the next, but the root does not, and
    ``w`` stays in the latent coordinate frame. Float64, so the
    log-determinant is reliable.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim, dtype=torch.float64))
        self.register_buffer("W", torch.eye(dim, dtype=torch.float64))
        self.register_buffer("S", torch.eye(dim, dtype=torch.float64))
        self.register_buffer("logdet", torch.zeros((), dtype=torch.float64))

    @torch.no_grad()
    def refit(self, u: torch.Tensor) -> None:
        """Refit to latent samples ``u`` of shape (N, D)."""
        if len(u) < 2:
            return
        u = u.to(self.mean.device, torch.float64)
        d = u.shape[1]
        eye = torch.eye(d, dtype=u.dtype, device=u.device)
        cov = torch.cov(u.T).reshape(d, d) + 1e-12 * eye
        ev, V = torch.linalg.eigh(cov)
        ev = ev.clamp(min=1e-12)
        self.mean.copy_(u.mean(0))
        self.W.copy_((V / ev.sqrt()) @ V.T)
        self.S.copy_((V * ev.sqrt()) @ V.T)
        self.logdet.fill_(float(-0.5 * torch.log(ev).sum()))

    def whiten(self, u: torch.Tensor) -> torch.Tensor:
        return (u.to(self.W.dtype) - self.mean) @ self.W

    def unwhiten(self, w: torch.Tensor) -> torch.Tensor:
        return self.mean + w.to(self.S.dtype) @ self.S


class WhitenedFlow(nn.Module):
    """A velocity field in the frame of its own whitener; samples and densities in the latent frame."""

    def __init__(self, param_dim: int, cond_dim: int, **net_kwargs):
        super().__init__()
        self.whitener = Whitener(param_dim)
        self.velocity = VelocityField(param_dim, cond_dim, **net_kwargs)

    @torch.no_grad()
    def sample(self, s: torch.Tensor, steps: int = 128, chunk: int = 16384) -> torch.Tensor:
        """One latent draw per row of ``s`` (float64)."""
        if len(s) == 0:
            return torch.zeros(0, self.velocity.param_dim, dtype=torch.float64, device=s.device)
        w = torch.cat([euler_sample(self.velocity, s[i:i + chunk], steps) for i in range(0, len(s), chunk)])
        return self.whitener.unwhiten(w)

    def log_prob(self, u: torch.Tensor, s: torch.Tensor, steps: int = 32, divergence: str = "exact",
                 n_probe: int = 4, chunk: int = 16384) -> torch.Tensor:
        """Latent log density at ``u`` (float64; -inf where the ODE ran away)."""
        if len(u) == 0:
            return torch.zeros(0, dtype=torch.float64, device=u.device)
        w = self.whitener.whiten(u).to(s.dtype)
        lp = torch.cat([
            cnf_logprob(self.velocity, w[i:i + chunk], s[i:i + chunk], steps, divergence, n_probe)
            for i in range(0, len(w), chunk)
        ])
        return torch.nan_to_num(lp.double() + self.whitener.logdet, nan=-math.inf)


def _robust_nll(log_prob: torch.Tensor, margin: float = 50.0) -> float:
    """Mean NLL without rows whose ODE ran away (non-finite, or ``margin`` nats below the median)."""
    ok = torch.isfinite(log_prob)
    if ok.any():
        ok &= log_prob > log_prob[ok].median() - margin
    return float(-log_prob[ok].mean()) if ok.any() else math.inf


# ==================== Estimator ====================


class FlowMatching(TorchModel):
    """Flow-matching posterior estimation with a truncated-prior proposal.

    Works in the standard-normal latent space of a ``TransformedPrior`` (e.g.
    ``Product``). Training runs in rounds (see ``RoundTrainer``); every round
    refits the whitener of each flow to the round's training data, except for
    a flow that was not promoted in the previous round, which keeps the frame
    it was trained in.

    Network groups: the conditional flow with the embedding (primary, judged
    on the validation NLL ``nll``) and the marginal flow (``nll_aux``). Both
    NLLs are latent-frame densities, comparable across whitener refits. Early
    stopping watches ``loss``, the conditional NLL without the rare rows whose
    ODE ran away. The training loss is the flow-matching loss. The evaluated,
    published and saved networks are exponential moving averages of the
    trained ones, and every round resumes training from them.

    The proposal is the prior truncated to the outer region of the region
    ladder (see ``falcon.estimators.region_ladder``), which moves once per
    round in the train actor. It needs the node's conditions at the
    observation; without them (e.g. amortized runs) it stays the prior.

    The defaults are the settings of the reference run that produced the O1b
    region of the LDC MBHB study (``t7b_s4k_b8k_max64k_reject_essgate``). Its
    buffer policy corresponds to ``discard_samples: true`` with
    ``buffer.min_samples`` as the floor, ``buffer.max_samples`` as the cap,
    ``buffer.simulate_when_full: false`` and ``buffer.validation_fraction:
    0.2``; its fixed 4000 simulations per round have no exact counterpart,
    since falcon simulates continuously.

    Args:
        max_epochs: Maximum epochs per round.
        lr: Learning rate; reset at the start of every round.
        embedding: Embedding config dict (with ``_target_`` etc.).
        device: Device string (e.g. ``"cuda:0"``); auto-detected if ``None``.
        batch_size: Mini-batch size.
        patience_epochs: End the round once the best validation loss is this
            many epochs old (checked at validations).
        val_every_epochs: Validate after every N-th epoch (and at the last
            epoch); validation solves the density ODE, so it is not cheap.
        max_rounds: Maximum number of rounds (``None`` = unlimited).
        patience_rounds: Stop training after this many rejected rounds in a row.
        prior_rounds: Rounds that simulate from the prior before switching to
            the learned proposal.
        cache_on_device: Cache training data on the estimator device.
        max_cache_samples: Cap on cached training samples (0 = all).
        discard_samples: After accepted rounds, discard samples outside the
            outer region (oldest first, never below ``buffer.min_samples``).
        hidden: Width of the velocity-field MLP.
        layers: Hidden layers of the velocity-field MLP.
        time_dim: Number of Fourier time features (even).
        layernorm: LayerNorm after every hidden layer.
        betas: AdamW beta coefficients.
        grad_clip: Gradient-norm clip, separately for the conditional flow
            with the embedding and for the marginal flow (0 = off).
        lr_decay_factor: LR decay factor for the plateau scheduler (1.0 = no decay).
        lr_patience_epochs: Epochs without validation improvement before LR decay.
        ema_decay: Decay of the exponential moving averages, per step.
        embedding_epochs: Train the embedding only in the first N epochs of
            each round, then keep it fixed (``None`` = train it throughout).
        time_late_k: Exponent of the late-time tilt of the training times.
        time_late_mix: Fraction of training times drawn from the tilt.
        sample_steps: Euler steps when drawing from a flow.
        density_steps: ODE steps when evaluating a density.
        div_probes: Hutchinson probes for the divergence in proposal and
            posterior densities (0 = exact trace); validation always uses the
            exact trace.
        eval_chunk: Rows per forward pass when sampling or evaluating densities.
        x_sigma: The inner region must contain this many sigmas of the
            conditional flow's mass (as two-sided normal tail mass: 3 means
            2.7e-3 may lie outside); also the posterior truncation.
        delta_sigma: Half-width of the ladder's dead band; new inner regions
            are minted at ``x_sigma + delta_sigma``.
        min_keep_frac: At least this fraction of a region's own marginal-flow
            draws must fall inside it.
        vratio: Contract only if the candidate has at most this fraction of
            the inner region's prior mass.
        v_min_ess: Effective sample size the mass ratio must carry to contract.
        v_max_draws: Maximum marginal-flow draws spent on that ratio.
        chain_depth: Ancestors tested when checking membership in a region
            (-1 = all).
        n_region: Draws per pass when minting or sampling a region.
        n_mout: Draws for the mass outside the inner region.
        max_sample_passes: Maximum passes when filling a proposal request.
        ess_factor: Extend the pool of weighted region draws until its
            effective sample size is this many times the draws taken from it.
        readout_draws: Draws used to set the posterior truncation.
    """

    def __init__(
        self,
        *,
        # Most commonly changed
        max_epochs: int = 200,
        lr: float = 3e-4,
        embedding=None,
        device: Optional[str] = None,
        # Training loop
        batch_size: int = 128,
        patience_epochs: int = 12,
        val_every_epochs: int = 3,
        max_rounds: Optional[int] = None,
        patience_rounds: int = 10,
        prior_rounds: int = 0,
        cache_on_device: bool = False,
        max_cache_samples: int = 0,
        discard_samples: bool = True,
        # Network
        hidden: int = 512,
        layers: int = 6,
        time_dim: int = 32,
        layernorm: bool = False,
        # Optimizer
        betas: tuple = (0.9, 0.9),
        grad_clip: float = 1.0,
        lr_decay_factor: float = 1.0,
        lr_patience_epochs: int = 8,
        ema_decay: float = 0.995,
        embedding_epochs: Optional[int] = 10,
        # Flow matching
        time_late_k: float = 8.0,
        time_late_mix: float = 0.5,
        sample_steps: int = 128,
        density_steps: int = 32,
        div_probes: int = 4,
        eval_chunk: int = 16384,
        # Region ladder
        x_sigma: float = 3.0,
        delta_sigma: float = 1.0,
        min_keep_frac: float = 0.01,
        vratio: float = 0.8,
        v_min_ess: float = 1000,
        v_max_draws: int = 1048576,
        chain_depth: int = 1,
        n_region: int = 65536,
        n_mout: int = 65536,
        max_sample_passes: int = 64,
        ess_factor: float = 4.0,
        readout_draws: int = 65536,
    ):
        if embedding_epochs is not None and embedding_epochs < 0:
            raise ValueError(f"embedding_epochs must be None or >= 0, got {embedding_epochs}")
        self.max_epochs = max_epochs
        self.lr = lr
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
        self.discard_samples = discard_samples
        self.hidden = hidden
        self.layers = layers
        self.time_dim = time_dim
        self.layernorm = layernorm
        self.betas = betas
        self.grad_clip = grad_clip
        self.lr_decay_factor = lr_decay_factor
        self.lr_patience_epochs = lr_patience_epochs
        self.ema_decay = ema_decay
        self.embedding_epochs = embedding_epochs
        self.time_late_k = time_late_k
        self.time_late_mix = time_late_mix
        self.sample_steps = sample_steps
        self.density_steps = density_steps
        self.div_probes = div_probes
        self.eval_chunk = eval_chunk
        self.x_sigma = x_sigma
        self.delta_sigma = delta_sigma
        self.min_keep_frac = min_keep_frac
        self.vratio = vratio
        self.v_min_ess = v_min_ess
        self.v_max_draws = v_max_draws
        self.chain_depth = chain_depth
        self.n_region = n_region
        self.n_mout = n_mout
        self.max_sample_passes = max_sample_passes
        self.ess_factor = ess_factor
        self.readout_draws = readout_draws

    def setup(self, simulator_instance, theta_key=None, condition_keys=None):
        if not isinstance(simulator_instance, TransformedPrior):
            raise TypeError(
                f"FlowMatching requires a TransformedPrior (e.g., Product), "
                f"got {type(simulator_instance).__name__}."
            )
        super().setup(simulator_instance, theta_key, condition_keys)
        self.param_dim = simulator_instance.param_dim

        self._embedding = None        # EMA embedding (state)
        self._flow_c = None           # EMA conditional flow (state)
        self._flow_m = None           # EMA marginal flow (state)
        self._cond_dim = None
        self._frozen_params = set()   # embedding parameters that the config keeps fixed
        self._train_nets = None       # trained copies, created in the train actor
        self._optimizer = None
        self._scheduler = None

        self._hold_gauge: Dict[str, bool] = {}
        self._steps = 0
        self._steps_per_epoch = 0
        self._observation = None
        self._warned_no_observation = False
        self._state_version = 0
        self._readout_thresholds: Dict[tuple, float] = {}

        ladder_config = LadderConfig(
            x_sigma=self.x_sigma, delta_sigma=self.delta_sigma, min_keep_frac=self.min_keep_frac,
            vratio=self.vratio, v_min_ess=self.v_min_ess, v_max_draws=self.v_max_draws,
            chain_depth=self.chain_depth, n_region=self.n_region, n_mout=self.n_mout,
            max_sample_passes=self.max_sample_passes, ess_factor=self.ess_factor,
        )
        self._ladder = RegionLadder(ladder_config, self._load_region, self.param_dim, self.device)

    # ==================== Networks ====================

    def init_from_batch(self, batch):
        return {
            "theta": self._to_array(batch[f"{self.theta_key}.value"]),
            "conditions": {
                k: self._to_array(batch[f"{k}.value"])
                for k in self.condition_keys if f"{k}.value" in batch
            },
        }

    def build(self, init_tree) -> None:
        debug("Initializing networks...")
        conditions = {k: self._to_tensor(v, self.device) for k, v in init_tree["conditions"].items()}
        embedding = instantiate_embedding(self.embedding).to(self.device)
        embedding.eval()
        with torch.no_grad():
            self._cond_dim = embedding(conditions).shape[1]
        self._frozen_params = {name for name, p in embedding.named_parameters() if not p.requires_grad}

        self._embedding = embedding
        self._flow_c = self._new_flow()
        self._flow_m = self._new_flow()
        for module in (self._embedding, self._flow_c, self._flow_m):
            module.requires_grad_(False)
        self._set_groups({
            "conditional": NetworkGroup(
                {"flow": self._flow_c, "embedding": self._embedding}, "nll",
                embedding=self._embedding,
            ),
            "marginal": NetworkGroup({"flow": self._flow_m}, "nll_aux"),
        })
        debug(f"FlowMatching networks built: param_dim={self.param_dim}, cond_dim={self._cond_dim}")

    def _new_flow(self) -> WhitenedFlow:
        return WhitenedFlow(
            self.param_dim, self._cond_dim, hidden=self.hidden, layers=self.layers,
            time_dim=self.time_dim, layernorm=self.layernorm,
        ).to(self.device)

    def import_state(self, group: str, tree) -> None:
        super().import_state(group, tree)
        self._state_version += 1
        self._readout_thresholds.clear()

    # ==================== Training ====================

    def _ensure_training(self) -> None:
        """Create the trained copies of the networks and the optimizer (train actor only)."""
        if self._train_nets is not None:
            return
        nets = {
            "velocity_c": copy.deepcopy(self._flow_c.velocity),
            "velocity_m": copy.deepcopy(self._flow_m.velocity),
            "embedding": copy.deepcopy(self._embedding),
        }
        for net in nets.values():
            net.requires_grad_(True)
        for name, p in nets["embedding"].named_parameters():
            if name in self._frozen_params:
                p.requires_grad_(False)
        self._train_nets = nets
        self._params_c = [p for net in (nets["velocity_c"], nets["embedding"])
                          for p in net.parameters() if p.requires_grad]
        self._params_m = list(nets["velocity_m"].parameters())
        self._optimizer = AdamW(
            [{"params": self._params_c}, {"params": self._params_m}], lr=self.lr, betas=tuple(self.betas),
        )
        self._build_scheduler()

    def _build_scheduler(self) -> None:
        # Stepped once per validation, so the patience is converted from epochs
        self._scheduler = (
            ReduceLROnPlateau(
                self._optimizer, mode="min",
                factor=self.lr_decay_factor,
                patience=-(-self.lr_patience_epochs // self.val_every_epochs),
            )
            if self.lr_decay_factor < 1.0 else None
        )

    def _latent(self, batch) -> torch.Tensor:
        theta = self._to_tensor(batch[f"{self.theta_key}.value"], self.device)
        return self.simulator_instance.inverse(theta, mode="standard_normal").double()

    def _conditions(self, batch) -> Dict[str, torch.Tensor]:
        return {
            k: self._to_tensor(batch[f"{k}.value"], self.device)
            for k in self.condition_keys if f"{k}.value" in batch
        }

    def _embedding_trains(self) -> bool:
        if self.embedding_epochs is None or self._steps_per_epoch == 0:
            return True
        return self._steps < self.embedding_epochs * self._steps_per_epoch

    def on_round_start(self) -> None:
        if self._train_nets is None:
            return
        # Resume training from the (EMA) best networks
        self._train_nets["velocity_c"].load_state_dict(self._flow_c.velocity.state_dict())
        self._train_nets["velocity_m"].load_state_dict(self._flow_m.velocity.state_dict())
        self._train_nets["embedding"].load_state_dict(self._embedding.state_dict())
        for group in self._optimizer.param_groups:
            group["lr"] = self.lr
        self._build_scheduler()

    def on_train_start(self, batches) -> None:
        self._ensure_training()
        u = torch.cat([self._latent(batch).cpu() for batch in batches])
        refit = []
        for name, flow in (("conditional", self._flow_c), ("marginal", self._flow_m)):
            if not self._hold_gauge.get(name, False):
                flow.whitener.refit(u)
                refit.append(name)
        held = [name for name in ("conditional", "marginal") if name not in refit]
        info(
            f"Whitener refit on {len(u)} samples"
            + (f" (frame held: {', '.join(held)})" if held else "")
            + f" | logdet conditional={float(self._flow_c.whitener.logdet):.3f}"
            f" marginal={float(self._flow_m.whitener.logdet):.3f}"
        )
        log({
            "whitener:logdet_conditional": float(self._flow_c.whitener.logdet),
            "whitener:logdet_marginal": float(self._flow_m.whitener.logdet),
        })
        self._steps = 0
        self._steps_per_epoch = len(u) // self.batch_size

    def train_step(self, batch) -> Dict[str, float]:
        self._ensure_training()
        nets = self._train_nets
        u = self._latent(batch)
        conditions = self._conditions(batch)
        w_c = self._flow_c.whitener.whiten(u).float()
        w_m = self._flow_m.whitener.whiten(u).float()

        joint = self._embedding_trains()
        if joint:
            nets["embedding"].train(True)
            s = nets["embedding"](conditions).float()
        else:
            with torch.no_grad():
                s = self._summary("conditional", conditions).float()

        loss_c = fm_loss_late(nets["velocity_c"], w_c, s, self.time_late_k, self.time_late_mix)
        loss_m = fm_loss_late(nets["velocity_m"], w_m, torch.zeros_like(s), self.time_late_k, self.time_late_mix)
        self._optimizer.zero_grad(set_to_none=True)
        (loss_c + loss_m).backward()
        if self.grad_clip:
            torch.nn.utils.clip_grad_norm_(self._params_c, self.grad_clip)
            torch.nn.utils.clip_grad_norm_(self._params_m, self.grad_clip)
        self._optimizer.step()

        ema_update(self._flow_c.velocity, nets["velocity_c"], self.ema_decay)
        ema_update(self._flow_m.velocity, nets["velocity_m"], self.ema_decay)
        if joint:
            ema_update(self._embedding, nets["embedding"], self.ema_decay)
        self._steps += 1
        return {"loss": loss_c.item(), "loss_aux": loss_m.item()}

    def val_step(self, batch) -> Dict[str, float]:
        u = self._latent(batch)
        s = self._summary("conditional", self._conditions(batch)).float()
        zeros = torch.zeros_like(s)
        kw = self._density_kw(exact=True)
        log_prob_c = self._flow_c.log_prob(u, s, **kw)
        log_prob_m = self._flow_m.log_prob(u, zeros, **kw)
        return {
            "loss": _robust_nll(log_prob_c),
            "nll": float(-log_prob_c.mean()),
            "nll_aux": float(-log_prob_m.mean()),
            "fm": val_fm_loss(self._flow_c.velocity, self._flow_c.whitener.whiten(u).float(), s),
            "fm_aux": val_fm_loss(self._flow_m.velocity, self._flow_m.whitener.whiten(u).float(), zeros),
        }

    def on_validation_end(self, epoch: int, val_metrics: Dict[str, float]) -> Optional[Dict[str, float]]:
        if self._scheduler is not None:
            self._scheduler.step(val_metrics.get("loss", float("inf")))
        lr = self._optimizer.param_groups[0]["lr"]
        log({"lr": lr})
        return {"lr": lr, "val_fm": val_metrics.get("fm", float("nan"))}

    def discard_test(self, batch):
        return ~self._ladder.cut(self._ladder.outer, self._latent(batch))

    # ==================== Proposal ====================

    def set_observations(self, conditions) -> None:
        self._observation = {k: np.asarray(v) for k, v in conditions.items()}

    def on_round_end(self, promoted: Dict[str, bool]) -> bool:
        # A flow that was not promoted resumes from the best network next
        # round, in the frame that network was trained in
        self._hold_gauge = {name: not p for name, p in promoted.items()}
        if self._observation is None:
            if not self._warned_no_observation:
                warning(
                    f"The conditions of '{self.theta_key}' are not fixed by the observations, "
                    "so the proposal stays the prior"
                )
                self._warned_no_observation = True
            return False
        conditions = {k: self._to_tensor(v, self.device) for k, v in self._observation.items()}
        with torch.no_grad():
            s_obs = self._summary("conditional", conditions).float()[:1]
        current = FlowPair(self._flow_c, self._flow_m, s_obs, self._sample_kw(), self._density_kw())
        trees = {
            "c": module_state(self._flow_c),
            "m": module_state(self._flow_m),
            "s_obs": s_obs.cpu().numpy(),
        }
        action = self._ladder.step(current, trees)
        return action in ("EXPAND", "CONTRACT")

    def export_proposal(self, full: bool):
        return self._ladder.export(full)

    def import_proposal(self, tree) -> None:
        self._ladder.load(tree)

    def _sample_kw(self) -> dict:
        return {"steps": self.sample_steps, "chunk": self.eval_chunk}

    def _density_kw(self, exact: bool = False) -> dict:
        kw = {"steps": self.density_steps, "chunk": self.eval_chunk}
        if exact or self.div_probes == 0:
            return {**kw, "divergence": "exact"}
        return {**kw, "divergence": "hutchinson", "n_probe": self.div_probes}

    def _load_region(self, record) -> FlowPair:
        flows = []
        for key in ("c", "m"):
            flow = self._new_flow()
            load_module_state(flow, record[key], self.device)
            flow.requires_grad_(False)
            flows.append(flow)
        s_obs = self._to_tensor(record["s_obs"], self.device).float()  # copies: Ray arrays are read-only
        return FlowPair(flows[0], flows[1], s_obs, self._sample_kw(), self._density_kw())

    # ==================== Sampling ====================

    def _sample_prior(self, num_samples: int) -> dict:
        samples = self.simulator_instance.simulate_batch(num_samples)
        u = self.simulator_instance.inverse(self._to_tensor(samples), mode="standard_normal")
        return {"value": samples, "log_prob": self._to_array(log_normal(u))}

    def _sample(self, num_samples: int, conditions, mode: str) -> dict:
        if mode == "proposal":
            u, log_prob = self._ladder.sample(num_samples)
        else:
            u, log_prob = self._posterior(num_samples, conditions)
        value = self.simulator_instance.forward(u, mode="standard_normal")
        result = {"value": self._to_array(value), "log_prob": self._to_array(log_prob)}
        if mode == "proposal":
            log({
                "sample_proposal:mean": float(result["value"].mean()),
                "sample_proposal:std": float(result["value"].std()),
                "sample_proposal:logprob": float(result["log_prob"].mean()),
            })
        return result

    def _posterior(self, num_samples: int, conditions):
        """Posterior draws, one readout per distinct observation.

        Rows with identical conditions share a readout: evidence derived from
        one broadcast observation arrives as ``num_samples`` identical rows.
        """
        arrays = {k: np.asarray(v) for k, v in conditions.items()}
        rows = max(len(a) for a in arrays.values())
        if rows == 1:
            return self._readout(num_samples, arrays)
        if rows != num_samples:
            raise ValueError(f"Conditions have {rows} rows for {num_samples} samples")
        groups: Dict[bytes, list] = {}
        for i in range(rows):
            key = b"".join(np.ascontiguousarray(a[i if len(a) > 1 else 0]).tobytes()
                           for _, a in sorted(arrays.items()))
            groups.setdefault(key, []).append(i)
        u = torch.empty(rows, self.param_dim, dtype=torch.float64, device=self.device)
        log_prob = torch.empty(rows, dtype=torch.float64, device=self.device)
        for idx in groups.values():
            first = idx[0]
            observation = {k: a[first:first + 1] if len(a) > 1 else a for k, a in arrays.items()}
            rows_idx = torch.as_tensor(idx, device=self.device)
            u[rows_idx], log_prob[rows_idx] = self._readout(len(idx), observation)
        return u, log_prob

    def _readout(self, num_samples: int, conditions):
        """Conditional-flow draws cut at the ``x_sigma`` contour, for one observation."""
        tensors = {k: self._to_tensor(v, self.device) for k, v in conditions.items()}
        s = self._summary("conditional", tensors).float()
        thr = self._readout_threshold(s, conditions)
        keep_frac = 1.0 - leak(self.x_sigma)
        us, log_probs, kept = [], [], 0
        for _ in range(self.max_sample_passes):
            if kept >= num_samples:
                break
            m = int(math.ceil(1.1 * (num_samples - kept) / keep_frac)) + 16
            u = self._flow_c.sample(s.expand(m, -1), **self._sample_kw())
            log_prob = self._flow_c.log_prob(u, s.expand(m, -1), **self._density_kw())
            keep = torch.isfinite(u).all(1) & (log_prob > thr)
            us.append(u[keep])
            log_probs.append(log_prob[keep])
            kept += int(keep.sum())
        if kept < num_samples:
            raise RuntimeError(f"Only {kept}/{num_samples} posterior draws above the truncation")
        return torch.cat(us)[:num_samples], torch.cat(log_probs)[:num_samples]

    def _readout_threshold(self, s: torch.Tensor, conditions) -> float:
        """The ``leak(x_sigma)`` quantile of ``ln q_c`` under its own draws, per network and observation."""
        digest = hashlib.sha1()
        for k in sorted(conditions):
            digest.update(k.encode())
            digest.update(np.ascontiguousarray(conditions[k]).tobytes())
        key = (self._state_version, digest.hexdigest())
        if key not in self._readout_thresholds:
            m = self.readout_draws
            # Own random stream: the threshold does not depend on, or advance, the caller's
            with self._seeded(np.random.default_rng(0)):
                u = self._flow_c.sample(s.expand(m, -1), **self._sample_kw())
                log_prob = self._flow_c.log_prob(u, s.expand(m, -1), **self._density_kw())
            log_prob = log_prob[torch.isfinite(log_prob)]
            if len(self._readout_thresholds) > 64:
                self._readout_thresholds.clear()
            self._readout_thresholds[key] = float(torch.quantile(log_prob, leak(self.x_sigma)))
        return self._readout_thresholds[key]
