"""Gaussianized flow-matching posterior estimator.

A deliberately simple, dynamic-SBI-friendly design:

  * A **single global** mean ``mu`` and covariance ``Sigma`` of the current training
    distribution (the prior's standard-normal latent ``theta_lat``), tracked by a plain
    EMA -- no conditional MLP, so there is nothing to over-fit per-observation. The
    whitening ``w = Sigma^{-1/2}(theta_lat - mu)`` standardises the buffer to ~N(0,I);
    its eigenvalues are clamped at the prior variance (= 1) so the posterior support can
    never inflate beyond the prior. In sequential / dynamic SBI the buffer concentrates
    each round, so this global whitening *zooms* automatically.

  * A **single** flow-matching velocity field that learns the residual structure in the
    whitened space, used both **conditionally** (on the embedding ``s = E(x)``) and
    **unconditionally** (on a learnable NULL token) -- classifier-free style. The
    conditional density is the posterior proposal, the null density is the marginal.

Sampling / density feed the same importance-sampling machinery as ``Flow``, adapted to
the standard-normal latent space: the latent prior is N(0, I), so log N(0,I) is added to
the importance weights, and a prior-width truncation backstops runaway-wide proposals.
A second, log-density truncation guards the *proposal* against the flow's poorly-modelled
tails: an EMA-estimated floor ``_logp_cond_floor`` (the ``truncation_alpha``-quantile of
``log_prob_cond`` under the posterior) penalises proposal candidates below it, so the
proposal never emits from the extreme low-density tail. Posterior sampling is untouched.
"""

import copy
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from falcon.core.logger import log, debug, info
from falcon.priors.product import TransformedPrior
from falcon.estimators.stepwise_base import StepwiseEstimator
from falcon.estimators.flow_matching import (
    VelocityField, EMA, fm_loss, euler_sample, cnf_logprob,
)
from falcon.embeddings import instantiate_embedding


class _GlobalWhitener(nn.Module):
    """Single global mean/covariance of ``theta_lat``, EMA-tracked.

    Whitening uses the **symmetric (ZCA) square root** ``w = (theta_lat - mu) @ Sigma^{-1/2}``
    with ``Sigma^{-1/2} = V diag(lambda^{-1/2}) V^T``. This stays in the original coordinate
    frame and is a function of ``Sigma`` alone, so eigenvector jitter / sign flips in
    (near-)degenerate subspaces cancel -- unlike the PCA whitening ``(.) @ V / sqrt(lambda)``
    (rotates into a jittering eigenframe -> the flow's target spins and smears to a blob)
    or Cholesky (stable per step, but its triangular factor ties the whitening directions
    to an arbitrary axis order, so they swing as correlations drift).

    The eigenvalues are clamped to ``[min_var, 1]`` -- the upper clamp at the prior variance
    keeps the posterior support inside the prior, the lower clamp bounds the zoom. There is
    no conditioning and no MLP, so the whitener cannot memorise samples -- it only tracks
    the buffer scale.
    """

    def __init__(self, param_dim: int, momentum: float, min_var: float, eig_update_freq: int):
        super().__init__()
        self.param_dim = param_dim
        self.momentum = momentum
        self.min_var = min_var
        self.eig_update_freq = eig_update_freq
        self._step = 0
        self.register_buffer("_mean", torch.zeros(param_dim, dtype=torch.float64))
        self.register_buffer("_cov", torch.eye(param_dim, dtype=torch.float64))
        self.register_buffer("_eigvals", torch.ones(param_dim, dtype=torch.float64))
        self.register_buffer("_inv_sqrt", torch.eye(param_dim, dtype=torch.float64))   # Sigma^{-1/2}
        self.register_buffer("_sqrt", torch.eye(param_dim, dtype=torch.float64))       # Sigma^{1/2}

    def to(self, *args, **kwargs):
        """Move module, preserving the float64 statistics buffers."""
        result = super().to(*args, **kwargs)
        for name in ("_mean", "_cov", "_eigvals", "_inv_sqrt", "_sqrt"):
            setattr(result, name, getattr(result, name).to(torch.float64))
        return result

    @torch.no_grad()
    def update(self, theta_lat: torch.Tensor):
        """EMA-update the frame; return the affine frame delta ``(M, c)`` or ``None``.

        The delta maps OLD whitened coordinates to NEW ones: ``w_new = w_old @ M + c``
        with ``M = S_old @ W_new`` and ``c = (mu_old - mu_new) @ W_new`` (row-vector
        convention, ``S = Sigma^{1/2}``, ``W = Sigma^{-1/2}``). Between eigen refreshes
        only the mean moves, so ``M = I`` and the delta is a pure shift. Returns
        ``None`` when the frame did not measurably change. Callers that fold the
        frame into a network (``whiten_fold``) consume this; everyone else may
        ignore the return value.
        """
        mu0 = self._mean.clone()
        S0 = self._sqrt.clone()
        m = self.momentum
        x = theta_lat.to(self._mean.dtype)
        self._mean = (1 - m) * self._mean + m * x.mean(dim=0)
        centered = x - self._mean
        n = x.shape[0]
        eye = torch.eye(self.param_dim, device=x.device, dtype=x.dtype)
        batch_cov = (centered.T @ centered) / max(n - 1, 1) + self.min_var * eye
        self._cov = (1 - m) * self._cov + m * batch_cov
        self._step += 1
        if self._step % self.eig_update_freq == 0:
            self._refresh()
        M = S0 @ self._inv_sqrt
        c = (mu0 - self._mean) @ self._inv_sqrt
        eye64 = torch.eye(self.param_dim, device=M.device, dtype=M.dtype)
        if (M - eye64).abs().max() < 1e-12 and c.abs().max() < 1e-12:
            return None
        return M, c

    def _refresh(self) -> None:
        eigvals, eigvecs = torch.linalg.eigh(self._cov)
        eigvals = eigvals.clamp(min=self.min_var, max=1.0)
        self._eigvals = eigvals
        self._inv_sqrt = (eigvecs / eigvals.sqrt()) @ eigvecs.T   # V diag(1/sqrt) V^T
        self._sqrt = (eigvecs * eigvals.sqrt()) @ eigvecs.T       # V diag(sqrt)   V^T

    def whiten(self, theta_lat: torch.Tensor) -> torch.Tensor:
        return (theta_lat.to(self._inv_sqrt.dtype) - self._mean) @ self._inv_sqrt

    def unwhiten(self, w: torch.Tensor) -> torch.Tensor:
        return self._mean + w.to(self._sqrt.dtype) @ self._sqrt

    def logdet(self) -> torch.Tensor:
        """log|d w / d theta_lat| folded into log_prob: -1/2 sum log lambda."""
        return -0.5 * torch.log(self._eigvals).sum()


class _WhitenedFlow(nn.Module):
    """Global whitener + one flow-matching field (conditional via ``s``, marginal via NULL).

    FlowDensity-compatible surface for the importance sampler:
        training_loss(theta_lat, s) -> {"fm_cond", "fm_marg", "total"}
        sample(n, s)                -> (n, B, param) latent samples
        log_prob(theta_lat, s)      -> (N, B) latent log-density

    Pass the real embedding ``s`` for the conditional branch and ``null_cond(B)`` for the
    marginal branch. All densities/samples are over ``theta_lat``; the constant whitening
    Jacobian is folded into log_prob.
    """

    def __init__(self, param_dim: int, cond_dim: int, *,
                 momentum: float, min_var: float, eig_update_freq: int,
                 cond_momentum: Optional[float] = None,
                 flow_hidden: int, flow_layers: int, time_dim: int, ema_decay: float,
                 sample_steps: int, density_steps: int, divergence: str, n_probe: int,
                 eval_chunk: int, layernorm: bool = True, antithetic: bool = True,
                 w_embed_dim: int = 0, cond_embed_dim: int = 0,
                 per_param_nets: bool = False, cond_per_param: bool = False,
                 focus_dims=None, glu_cond: bool = False,
                 residual: bool = False, time_alpha: float = 0.0,
                 whiten_fold: bool = False, fold_threshold: float = 0.02):
        super().__init__()
        self.param_dim = param_dim
        self.cond_dim = cond_dim
        self.sample_steps = sample_steps
        self.density_steps = density_steps
        self.divergence = divergence
        self.n_probe = n_probe
        self.eval_chunk = eval_chunk
        self.antithetic = antithetic
        self.time_alpha = time_alpha
        self.whiten_fold = whiten_fold
        self.fold_threshold = fold_threshold
        if whiten_fold:
            # The fold rewrites the first/last Linear of a shared trunk where w
            # occupies the leading param_dim input columns and the output layer
            # emits the velocity directly. Reject architectures that break either
            # assumption (w entering elsewhere, or per-parameter heads).
            if per_param_nets:
                raise ValueError("whiten_fold requires a shared trunk (per_param_nets=False)")
            if w_embed_dim > 0:
                raise ValueError("whiten_fold requires w_embed_dim=0 (w must enter net[0] directly)")
            if glu_cond:
                raise ValueError("whiten_fold is incompatible with glu_cond (gates also consume w)")
            if focus_dims:
                raise ValueError("whiten_fold is incompatible with focus_dims")
            # Deadband accumulator: frame deltas compose affinely and are accumulated
            # here; the fold fires only when the ACCUMULATED delta crosses
            # fold_threshold. EMA estimation noise is mean-reverting, so on a
            # stationary buffer the accumulator hovers at identity and no fold
            # ever fires (exact baseline behavior); systematic drift (zoom,
            # branch-mass motion) adds up and is absorbed in function-preserving
            # folds. Per-step folding (threshold=0) injects the stats noise
            # straight into the weights - measured to cost ~2 nats val on a
            # stationary 32k buffer - so keep the threshold finite.
            self.register_buffer("_fold_Macc", torch.eye(param_dim, dtype=torch.float64))
            self.register_buffer("_fold_cacc", torch.zeros(param_dim, dtype=torch.float64))

        self.whitener = _GlobalWhitener(param_dim, momentum, min_var, eig_update_freq)
        self.velocity = VelocityField(param_dim, cond_dim, flow_hidden, flow_layers, time_dim, layernorm,
                                      w_embed_dim=w_embed_dim, cond_embed_dim=cond_embed_dim,
                                      per_param_nets=per_param_nets, cond_per_param=cond_per_param,
                                      focus_dims=focus_dims, glu_cond=glu_cond,
                                      residual=residual)
        self.velocity_ema = EMA.clone(self.velocity)
        self._ema = EMA(ema_decay)
        self.null_token = nn.Parameter(torch.zeros(cond_dim))   # learnable "no conditioning"

        # EMA mean/std of the summary, mirroring GaussianFullCov's _input_mean/_input_std:
        # keep the conditioning at ~unit scale so it neither swamps w nor buries the fine
        # x-dependence that pins the zoomed posterior. (null stays learnable in this space.)
        # Summary-normalisation EMA rate.  Decoupled from the whitener's momentum:
        # cond() re-gauges the network's AMPLITUDE sensitivity, and x_obs passes
        # through the same transform, so a fast rate makes the target's own
        # representation chase the buffer -- a feedback loop that lands on D_L,
        # the only pure-amplitude parameter.  At momentum 0.01 the horizon is ~100
        # steps (<2 epochs), i.e. it fully re-adapts within a rung.
        self.cond_momentum = momentum if cond_momentum is None else cond_momentum
        self.register_buffer("_cond_mean", torch.zeros(cond_dim))
        self.register_buffer("_cond_std", torch.ones(cond_dim))
        # When True, the summary normalisation stops tracking the buffer.  D_L is
        # the only PURE-AMPLITUDE parameter (h ~ 1/D_L); every other parameter is
        # carried by shape/phase.  A normalisation that re-gauges amplitude every
        # step therefore lands almost entirely on D_L -- and x_obs passes through
        # the same moving transform, so the target's representation drifts while
        # the data is fixed.  Measured: _cond_std contracts 5-10x over a run, and
        # the contraction rank-orders with the D_L bias (earwig 4.6x/+3.2sig,
        # husky 6.8x/+11.3sig, sponge 9.3x/+13.7sig).
        self.cond_stats_frozen = False

    # ---- EMA ----
    def ema_update(self) -> None:
        self._ema.update(self.velocity_ema, self.velocity)

    def null_cond(self, batch: int) -> torch.Tensor:
        return self.null_token[None].expand(batch, self.cond_dim)

    # ---- summary normalization (EMA diagonal whitening of the conditioning) ----
    @torch.no_grad()
    def _update_cond_stats(self, s: torch.Tensor) -> None:
        if self.cond_stats_frozen:
            return
        m = self.cond_momentum
        s = s.detach().to(self._cond_mean.dtype)
        self._cond_mean = (1 - m) * self._cond_mean + m * s.mean(dim=0)
        n = s.shape[0]
        var = ((s - self._cond_mean) ** 2).sum(dim=0) / max(n - 1, 1)
        self._cond_std = (1 - m) * self._cond_std + m * var.clamp(min=1e-12).sqrt()

    def cond(self, s: torch.Tensor) -> torch.Tensor:
        """Normalize the conditional summary to ~unit scale (frozen stats outside training)."""
        return (s.to(self._cond_mean.dtype) - self._cond_mean) / self._cond_std

    # ---- training ----
    @torch.no_grad()
    def _fold_frame_delta(self, M: torch.Tensor, c: torch.Tensor) -> None:
        """Fold an affine whitener-frame delta into the velocity nets.

        The whitener frame moved: old coordinates map to new ones as
        ``w_new = w_old @ M + c``. The function-preserving warm start is the
        target-side conjugation ``v_new(w) = v_old((w - c) @ M^{-1}) @ M``,
        absorbed into weights (PyTorch Linear: ``out = x @ weight.T + bias``):

          input side  (w = first ``P`` columns of ``net[0]``):
            cols  <- cols @ inv(M).T
            bias  <- bias - (c @ inv(M)) @ cols_old.T
          output side (``net[-1]``):
            weight <- M.T @ weight
            bias   <- bias @ M

        Applied identically to the live and EMA nets (both transforms are
        affine in the parameters, so EMA-of-fold == fold-of-EMA and the pair
        stays consistent). Optimizer moments are deliberately untouched: the
        per-step delta is O(momentum), the moment mismatch second order.
        The base-distribution mismatch of the conjugation is likewise
        O(||M - I||) per step (see docs/PLAN_frame_folding.md).
        """
        P = self.param_dim
        Minv = torch.linalg.inv(M)               # float64
        shift = c @ Minv                         # (P,)
        for field in (self.velocity, self.velocity_ema):
            first, last = field.net[0], field.net[-1]
            dt = first.weight.dtype
            cols_old = first.weight[:, :P].to(M.dtype)
            first.bias.data.sub_((shift @ cols_old.T).to(dt))
            first.weight.data[:, :P].copy_((cols_old @ Minv.T).to(dt))
            last.weight.data.copy_((M.T @ last.weight.to(M.dtype)).to(dt))
            last.bias.data.copy_((last.bias.to(M.dtype) @ M).to(dt))

    def training_loss(self, theta_lat: torch.Tensor, s: torch.Tensor) -> Dict[str, torch.Tensor]:
        if self.training:
            delta = self.whitener.update(theta_lat)
            if self.whiten_fold and delta is not None:
                M, c = delta
                # compose onto the accumulator: w2 = (w0 @ Ma + ca) @ M + c
                self._fold_cacc = self._fold_cacc @ M + c
                self._fold_Macc = self._fold_Macc @ M
                eye = torch.eye(self.param_dim, dtype=torch.float64,
                                device=self._fold_Macc.device)
                if max((self._fold_Macc - eye).abs().max().item(),
                       self._fold_cacc.abs().max().item()) > self.fold_threshold:
                    self._fold_frame_delta(self._fold_Macc, self._fold_cacc)
                    self._fold_Macc = eye
                    self._fold_cacc = torch.zeros_like(self._fold_cacc)
            self._update_cond_stats(s)
        w = self.whitener.whiten(theta_lat).detach().float()        # flow sees a fixed whitened target
        s_n = self.cond(s.float())
        null = self.null_token[None].expand(w.shape[0], self.cond_dim)
        fm_cond = fm_loss(self.velocity, w, s_n, self.antithetic,
                          time_alpha=self.time_alpha)
        fm_marg = fm_loss(self.velocity, w, null, self.antithetic,
                          time_alpha=self.time_alpha)
        return {"fm_cond": fm_cond, "fm_marg": fm_marg, "total": fm_cond + fm_marg}

    # ---- sampling: (n, B, param) ----  chunked over n*B to bound memory
    @torch.no_grad()
    def sample(self, n: int, s: torch.Tensor) -> torch.Tensor:
        B, C, P = s.shape[0], s.shape[1], self.param_dim
        s_flat = s[None].expand(n, B, C).reshape(n * B, C).float()
        outs = []
        for i in range(0, n * B, self.eval_chunk):
            sc = s_flat[i:i + self.eval_chunk]
            w = euler_sample(self.velocity_ema, sc, P, self.sample_steps)
            outs.append(self.whitener.unwhiten(w))
        return torch.cat(outs, 0).reshape(n, B, P)                   # (n, B, P) f64

    # ---- density: (N, B, param) -> (N, B) ----  chunked over N*B
    def log_prob(self, theta_lat: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        N, B, P = theta_lat.shape
        C = s.shape[1]
        w_all = self.whitener.whiten(theta_lat.reshape(N * B, P)).float()
        s_flat = s[None].expand(N, B, C).reshape(N * B, C).float()
        logdet = self.whitener.logdet()
        outs = []
        for i in range(0, N * B, self.eval_chunk):
            lp = cnf_logprob(self.velocity_ema, w_all[i:i + self.eval_chunk],
                             s_flat[i:i + self.eval_chunk],
                             self.density_steps, self.divergence, self.n_probe)
            outs.append(lp.to(self.whitener._mean.dtype) + logdet)
        return torch.cat(outs, 0).reshape(N, B)

    # ---- held-out NLL in theta_lat space (selection metric) ----
    @torch.no_grad()
    def val_nll(self, theta_lat: torch.Tensor, s: torch.Tensor,
                steps: int, n_probe: int) -> Dict[str, float]:
        """Per-pair NLL of theta_lat under the model, conditional (s) and marginal (null).

        ``log p(theta_lat | c) = cnf_logprob(whiten(theta_lat), c) + whitener.logdet()`` --
        a true density that *includes the whitening log-volume term* (``-1/2 sum log lambda``),
        so it falls as the buffer compresses, unlike the scale-invariant whitened FM loss.

        For *selection* the density only needs to rank checkpoints, so it uses cheap
        Hutchinson divergence (``n_probe``, noise averages over the val set) and a small
        ``steps`` (a consistent Euler bias cancels epoch-to-epoch) -- far cheaper than the
        exact/high-step density used for importance sampling.
        """
        M = theta_lat.shape[0]
        w = self.whitener.whiten(theta_lat).float()                         # (M, P)
        logdet = self.whitener.logdet()
        out_dtype = self.whitener._mean.dtype

        def nll(cond: torch.Tensor) -> float:
            lp = cnf_logprob(self.velocity_ema, w, cond.float(), steps, "hutch", n_probe)
            lp = lp.to(out_dtype) + logdet
            lp = torch.nan_to_num(lp, nan=-1e6, neginf=-1e6)
            return -lp.mean().item()

        return {"nll_cond": nll(self.cond(s)), "nll_marg": nll(self.null_cond(M))}


class GaussianizedFlowMatching(StepwiseEstimator):
    """Global-whitener + single-flow (cond/null) posterior estimator.

    Args:
        max_epochs, lr, gamma, embedding, device: as in other estimators.
        batch_size, early_stop_patience, prior_epochs, cache_*: training loop.
        momentum, min_var, eig_update_freq: global whitener EMA / eigendecomp.
        flow_hidden, flow_layers, time_dim, ema_decay: flow-matching velocity net.
        sample_steps: Euler steps for sampling.
        density_steps: Euler steps for the backward CNF density.
        divergence: "hutch" (Hutchinson, dimension-independent) or "exact" (d VJPs).
        n_probe: Hutchinson probe count (tunable; trades density noise vs cost).
        betas, lr_decay_factor, lr_patience, grad_clip: optimizer / scheduler (grad_clip<=0 disables clipping).
        discard_samples: deprecate buffer samples that fall below the proposal-truncation floor.
        use_best_models: use best-checkpoint networks (flow + whitener) for sampling.
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
        # rung training (see TrainingLoopConfig); 0 = off, legacy behaviour
        rung_epochs: int = 0,
        rung_min_epochs: int = 8,
        rung_max_epochs: int = 200,
        rung_patience: int = 12,
        rung_floor_refits: int = 5,  # proposal pools used to re-fit the floor per rung
        rung_best_settle_only: bool = True,  # no checkpointing in a rung's first rung_min_epochs
        cond_freeze_rung: int = 0,  # >0: freeze summary normalisation from this rung on (0 = never)
        # global whitener
        momentum: float = 0.01,
        cond_momentum: Optional[float] = None,  # None = share `momentum` (legacy)
        min_var: float = 1e-20,
        eig_update_freq: int = 1,
        whiten_fold: bool = False,  # fold whitener frame deltas into the velocity nets (docs/PLAN_frame_folding.md)
        fold_threshold: float = 0.02,  # deadband width: fold only when the accumulated delta exceeds this
        # flow-matching net
        flow_hidden: int = 256,
        flow_layers: int = 4,
        time_dim: int = 64,
        w_embed_dim: int = 0,     # >0: linear up-projection of w before the input concat
        cond_embed_dim: int = 0,  # >0: linear compression of the summary before the input concat
        per_param_nets: bool = False,  # one independent MLP per parameter instead of a shared trunk
        glu_cond: bool = False,    # FMPE-style (w,t) GLU gates on every hidden layer
        residual: bool = False,    # residual hidden blocks in the velocity trunk
        time_alpha: float = 0.0,   # power-law time prior (FMPE Sec 3.3); 0 = uniform
        cond_per_param: bool = False,  # leg i sees only summary slice i (pair with n_queries=param_dim)
        focus_dims=None,          # DEBUG: flow models only these free-param indices; rest = whitened Gaussian
        layernorm: bool = True,
        antithetic: bool = True,
        ema_decay: float = 0.9,
        sample_steps: int = 128,
        density_steps: int = 64,
        divergence: str = "hutch",
        n_probe: int = 4,
        eval_chunk: int = 50000,
        # cheap CNF density for the val-NLL selection metric (ranking only)
        val_density_steps: int = 16,
        val_n_probe: int = 1,
        # optimizer
        betas: tuple = (0.9, 0.9),
        lr_decay_factor: float = 1.0,
        lr_patience: int = 8,
        grad_clip: float = 1.0,
        # inference
        discard_samples: bool = False,
        use_best_models: bool = True,
        num_proposals: int = 256,
        proposal_mixture_beta: float = 0.5,
        prior_sigma_bound: float = 6.0,
        out_of_bounds_penalty: float = 100.0,
        tight_proposal_mode: bool = True,
        nan_replacement: float = -100.0,
        # proposal truncation via EMA-estimated log-density floor (alpha=0 disables)
        truncation_alpha: float = 1e-6,      # ~5 sigma tail: fraction of posterior mass below the floor
        truncation_momentum: float = 0.05,   # EMA rate for the floor
        truncation_warmup_epochs: int = 0,   # engage truncation only once total epochs >= this
        # plasticity: periodic shrink-and-perturb of the live velocity head (EMA untouched)
        plasticity_period: int = 0,      # 0 = off; else apply every N epochs
        plasticity_shrink: float = 0.9,  # lambda in  w <- lam*w + (1-lam)*fresh_init
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
        self.rung_epochs = rung_epochs
        self.rung_min_epochs = rung_min_epochs
        self.rung_max_epochs = rung_max_epochs
        self.rung_patience = rung_patience
        self.rung_floor_refits = rung_floor_refits
        self.rung_best_settle_only = rung_best_settle_only
        self.cond_freeze_rung = cond_freeze_rung
        self.momentum = momentum
        self.cond_momentum = cond_momentum
        self.min_var = min_var
        self.eig_update_freq = eig_update_freq
        self.whiten_fold = whiten_fold
        self.fold_threshold = fold_threshold
        self.flow_hidden = flow_hidden
        self.flow_layers = flow_layers
        self.time_dim = time_dim
        self.w_embed_dim = w_embed_dim
        self.cond_embed_dim = cond_embed_dim
        self.per_param_nets = per_param_nets
        self.glu_cond = glu_cond
        self.residual = residual
        self.time_alpha = time_alpha
        self.cond_per_param = cond_per_param
        self.focus_dims = focus_dims
        self.layernorm = layernorm
        self.antithetic = antithetic
        self.ema_decay = ema_decay
        self.sample_steps = sample_steps
        self.density_steps = density_steps
        self.divergence = divergence
        self.n_probe = n_probe
        self.eval_chunk = eval_chunk
        self.val_density_steps = val_density_steps
        self.val_n_probe = val_n_probe
        self.betas = betas
        self.lr_decay_factor = lr_decay_factor
        self.lr_patience = lr_patience
        self.grad_clip = grad_clip
        self.discard_samples = discard_samples
        self.use_best_models = use_best_models
        self.num_proposals = num_proposals
        self.proposal_mixture_beta = proposal_mixture_beta
        self.prior_sigma_bound = prior_sigma_bound
        self.out_of_bounds_penalty = out_of_bounds_penalty
        self.nan_replacement = nan_replacement
        self.plasticity_period = plasticity_period
        self.plasticity_shrink = plasticity_shrink
        self.tight_proposal_mode = tight_proposal_mode
        self.truncation_alpha = truncation_alpha
        self.truncation_momentum = truncation_momentum
        self.truncation_warmup_epochs = truncation_warmup_epochs

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

        self._flow = None
        self._best_flow = None
        self._best_embedding = None
        self._init_parameters = None
        self._best_loss = float("inf")
        self._optimizer = None
        self._scheduler = None
        self._logp_cond_floor = None   # EMA log_prob_cond alpha-quantile floor (proposal truncation)
        self._target_summary = None    # raw embedding of the TARGET obs (for buffer discard)
        # Rung mode: the floor is re-fitted only from a settled network, over a
        # bounded number of proposal pools at each rung boundary, then frozen.
        # None = unbounded (legacy: the EMA tracks every proposal call forever).
        self._floor_update_budget = None

    # ==================== Initialization ====================

    def _build_module(self, param_dim: int, cond_dim: int) -> _WhitenedFlow:
        return _WhitenedFlow(
            param_dim, cond_dim,
            momentum=self.momentum, min_var=self.min_var, eig_update_freq=self.eig_update_freq,
            cond_momentum=self.cond_momentum,
            flow_hidden=self.flow_hidden, flow_layers=self.flow_layers, time_dim=self.time_dim,
            ema_decay=self.ema_decay, sample_steps=self.sample_steps,
            density_steps=self.density_steps, divergence=self.divergence, n_probe=self.n_probe,
            eval_chunk=self.eval_chunk, layernorm=self.layernorm, antithetic=self.antithetic,
            w_embed_dim=self.w_embed_dim, cond_embed_dim=self.cond_embed_dim,
            glu_cond=self.glu_cond, residual=self.residual,
            time_alpha=self.time_alpha, whiten_fold=self.whiten_fold,
            fold_threshold=self.fold_threshold,
            per_param_nets=self.per_param_nets, cond_per_param=self.cond_per_param,
            focus_dims=self.focus_dims,
        ).to(self.device)

    def _initialize_networks(self, theta: torch.Tensor, conditions: Dict) -> None:
        debug("Initializing GaussianizedFlowMatching networks...")
        self._init_parameters = [theta, conditions]

        conditions_device = {k: v.to(self.device) for k, v in conditions.items()}
        s = self._embed(conditions_device, train=False).detach()
        theta_lat = self.simulator_instance.inverse(theta.to(self.device), mode="standard_normal")

        param_dim, cond_dim = theta_lat.shape[1], s.shape[1]
        self._flow = self._build_module(param_dim, cond_dim)
        self._best_flow = copy.deepcopy(self._flow)
        self._best_embedding = copy.deepcopy(self._embedding)

        params = [
            p for m in (self._embedding, self._flow)
            for p in m.parameters() if p.requires_grad
        ]
        self._optimizer = AdamW(params, lr=self.lr, betas=self.betas)
        self._scheduler = (
            ReduceLROnPlateau(
                self._optimizer, mode="min",
                factor=self.lr_decay_factor, patience=self.lr_patience,
            )
            if self.lr_decay_factor < 1.0 else None  # 1.0 = LR decay off
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

    def train_step(self, batch) -> Dict[str, float]:
        theta, _, conditions, theta_lat = self._unpack(batch, "train")
        if not self.networks_initialized:
            self._initialize_networks(theta, conditions)

        s = self._embed(conditions, train=True)
        self._flow.train()
        self._optimizer.zero_grad()
        losses = self._flow.training_loss(theta_lat, s)
        losses["total"].backward()
        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                (p for m in (self._embedding, self._flow) for p in m.parameters()),
                self.grad_clip)
        self._optimizer.step()
        self._flow.ema_update()

        if self.discard_samples:
            batch.discard(self._compute_discard_mask(theta_lat, s.detach()))

        # Train loss = the optimized FM objective (scale-invariant in the whitened space).
        return {"loss": losses["total"].item(),
                "fm_cond": losses["fm_cond"].item(), "fm_marg": losses["fm_marg"].item()}

    def val_step(self, batch) -> Dict[str, float]:
        _, _, conditions, theta_lat = self._unpack(batch, "val")
        s = self._embed(conditions, train=False)
        self._flow.eval()
        with torch.no_grad():
            fm = self._flow.training_loss(theta_lat, s)         # diagnostic only
            nll = self._flow.val_nll(theta_lat, s,              # selection metric (cheap CNF)
                                     self.val_density_steps, self.val_n_probe)
        # "loss" = total held-out NLL in theta_lat space (incl. whitening log-volume term);
        # this drives early-stopping (base loop) AND best-model selection (on_epoch_end),
        # so models are accepted as the density genuinely sharpens, not when FM wobbles.
        return {"loss": nll["nll_cond"] + nll["nll_marg"],
                "nll_cond": nll["nll_cond"], "nll_marg": nll["nll_marg"],
                "fm_cond": fm["fm_cond"].item(), "fm_marg": fm["fm_marg"].item()}

    def on_epoch_end(self, epoch: int, val_metrics: Dict[str, float]) -> Optional[Dict[str, float]]:
        val_loss = val_metrics.get("loss", float("inf"))        # total latent NLL

        # Rung mode: suppress checkpointing while the net is still settling into
        # this rung's data.  _best_loss was reset to inf at the boundary, so the
        # first epoch would otherwise always win and _best_flow -- which is the
        # DEPLOYED net, generating proposals and scoring discards -- would become
        # the least-settled net of the rung.  While settling, _best_flow stays the
        # previous rung's converged net: a frozen reference, and a self-consistent
        # one, since this rung's data was generated by exactly that net.
        settling = (
            self.rung_epochs > 0
            and self.rung_best_settle_only
            and self._epoch_in_rung < self.rung_min_epochs
        )
        if val_loss < self._best_loss and not settling:
            self._best_loss = val_loss
            self._best_flow.load_state_dict(self._flow.state_dict())
            self._best_embedding.load_state_dict(self._embedding.state_dict())
            log({"checkpoint": epoch})

        if self._scheduler is not None:
            self._scheduler.step(val_loss)

        # Plasticity: periodically soft-reset the live velocity head. The EMA/deployed
        # model is left intact, so the proposal stays stable while the live net re-engages.
        if self.plasticity_period > 0 and epoch > 0 and epoch % self.plasticity_period == 0:
            self._shrink_and_perturb()
            log({"plasticity:shrink_perturb": epoch})

        lr = self._optimizer.param_groups[0]["lr"]
        log({"lr": lr,
             "val_nll_cond": val_metrics.get("nll_cond", float("nan")),
             "val_nll_marg": val_metrics.get("nll_marg", float("nan")),
             "whiten_logdet": self._flow.whitener.logdet().item(),       # compression term
             "whiten_eigvals_mean": self._flow.whitener._eigvals.mean().item(),
             "logp_cond_floor": (self._logp_cond_floor
                                 if self._logp_cond_floor is not None else float("nan"))})
        return {"lr": lr}

    # ==================== Rung hooks ====================

    def on_rung_start(self, rung: int) -> None:
        """Reset the checkpoint high-water mark for the new rung.

        ``_best_loss`` is otherwise an all-time minimum, but the val NLL is not
        comparable across rungs: each rung validates against different frozen
        data, and the latent NLL carries the whitener's log-volume term, which
        shrinks as the dataset concentrates.  Left global, the checkpoint would
        latch onto whichever rung happened to have the most concentrated data
        and never update again.
        """
        self._best_loss = float("inf")
        if (self.cond_freeze_rung > 0 and rung >= self.cond_freeze_rung
                and self._flow is not None and not self._flow.cond_stats_frozen):
            self._flow.cond_stats_frozen = True
            log({"cond_stats_frozen_at_rung": rung})
            info(f"Summary normalisation FROZEN at rung {rung} "
                 f"(cond_std median {self._flow._cond_std.median().item():.4f})")

    def on_rung_end(self, rung: int, val_metrics: Dict[str, float]) -> None:
        """Re-open the density floor for re-fitting from the settled network."""
        self._floor_update_budget = self.rung_floor_refits
        log({"rung_end": rung,
             "rung_end_val_loss": val_metrics.get("loss", float("nan")),
             "logp_cond_floor": (self._logp_cond_floor
                                 if self._logp_cond_floor is not None else float("nan"))})

    @torch.no_grad()
    def _shrink_and_perturb(self) -> None:
        """w <- lam*w + (1-lam)*fresh_init on the live velocity MLP only.

        Restores plasticity without forgetting: only ``_flow.velocity`` is touched -- not
        the EMA (deployed model), the whitener, the embedding, or the null token -- so the
        proposal keeps sampling from the stable EMA while the live net regains plasticity
        and the EMA tracks it back over the next ``plasticity_period`` epochs.
        """
        lam = self.plasticity_shrink
        fresh = VelocityField(self._flow.param_dim, self._flow.cond_dim,
                              self.flow_hidden, self.flow_layers, self.time_dim, self.layernorm).to(self.device)
        for p, pf in zip(self._flow.velocity.parameters(), fresh.parameters()):
            p.mul_(lam).add_(pf.to(p.dtype), alpha=1 - lam)

    def _compute_discard_mask(self, theta_lat, s, p = 0.1):
        # Deprecate buffer samples in the proposal-truncated tail (log_prob_cond < floor).
        if self._logp_cond_floor is None or self._target_summary is None or np.random.rand(1) > p:
            return torch.zeros(len(theta_lat), dtype=torch.bool)
        self._best_flow.eval()
        with torch.no_grad():
            # condition on the TARGET obs (matches the floor), not the batch's own x_i;
            # [:1] guards against a multi-row stash of duplicated shared observations
            log_prob = self._best_flow.log_prob(
                theta_lat.unsqueeze(0), self._best_flow.cond(self._target_summary[:1])).squeeze(0).cpu()
        return log_prob < self._logp_cond_floor

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
        t_start = time.time()
        conditions = {k: self._to_tensor(v, self.device) for k, v in conditions.items()}
        # Evidence nodes re-simulated per-sample from a broadcast observation
        # (e.g. tokens derived from the shared observed x) arrive as num_samples
        # identical rows. Collapse them to batch 1 so the shared-observation
        # truncation logic applies and _target_summary is stored single-row
        # (the buffer-discard path expands it against arbitrary batch sizes).
        # Collapse jointly: derived evidence (e.g. tokens re-simulated from the
        # shared observed x by a stochastic/adaptive tokenizer) has non-identical
        # rows even though it represents the same observation. If ANY key
        # collapses, the batch is a shared-observation broadcast — take row 0 of
        # every key so multi-stream embeddings see a consistent batch size.
        # A batch-1 key is by construction a broadcast (single repeated ref),
        # so it witnesses a shared observation even when derived evidence rows
        # are not bitwise identical.
        # FIXME: taking row 0 of non-identical derived evidence assumes those
        # nodes are (effectively) deterministic functions of the observation.
        # For genuinely stochastic derived evidence this silently conditions on
        # one arbitrary realization; the proper fix is executor-level (compute
        # derived evidence once per observation and broadcast the ref).
        shared = any(
            v.shape[0] == 1 or torch.equal(v, v[:1].expand_as(v))
            for v in conditions.values()
        )
        if shared:
            conditions = {k: v[:1] for k, v in conditions.items()}

        use_best = self.use_best_models and self._best_flow is not None
        flow = self._best_flow if use_best else self._flow
        s = self._embed(conditions, train=False, use_best_fit=use_best).detach()
        self._target_summary = s                                  # stash target obs (for buffer discard)
        obs_batch = s.shape[0]                                     # observations before expand
        # Pooled SIR: with a single shared observation, all columns would be
        # statistically identical replicas — generate ONE candidate pool and
        # resample num_samples draws from it at the end, instead of num_samples
        # independent pools. Cuts ODE/CNF work by ~num_samples x per call.
        B = 1 if obs_batch == 1 else num_samples                   # internal column count
        s = s.expand(B, *s.shape[1:])                              # (B, C)
        s = flow.cond(s)                                          # normalize summary (frozen stats)
        null = flow.null_cond(B)                                  # (B, C)

        flow.eval()

        # Multiple-importance-sampling proposal mixture (balance heuristic), as in Flow.
        n_cond = max(0, min(self.num_proposals, int(round(self.proposal_mixture_beta * self.num_proposals))))
        n_marg = self.num_proposals - n_cond
        parts = []
        if n_cond > 0:
            parts.append(flow.sample(n_cond, s))
        if n_marg > 0:
            parts.append(flow.sample(n_marg, null))
        proposals = torch.cat(parts, dim=0)                       # (num_proposals, B, P)

        log_prob_cond = flow.log_prob(proposals, s)
        log_prob_marg = flow.log_prob(proposals, null)

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

        # Proposal truncation: EMA-estimate a log-density floor from the UNtruncated posterior
        # weights, then penalise proposal candidates below it. tau estimation must never see the
        # penalty (else it runs away), so `trunc` is applied only to the emitted-proposal weights.
        trunc = 0.0
        if self.truncation_alpha > 0.0:
            #assert obs_batch == 1, "proposal truncation assumes a single shared observation"
            # TODO: figure out what works best -- full posterior weights vs cond-only below
            #logw_post = (log_prob_cond - log_prob_marg + log_prior) - log_g_mix - mask
            logw_post = log_prob_cond - log_g_mix - mask
            logw_post = torch.nan_to_num(logw_post, nan=self.nan_replacement, neginf=self.nan_replacement)
            logw_post = logw_post - torch.logsumexp(logw_post, dim=0, keepdim=True)
            w_post = torch.exp(logw_post)                          # (num_proposals, B)

            if mode == "proposal":
                # EMA of the pooled weighted alpha-quantile of log_prob_cond (columns share one obs).
                # In rung mode the budget is refilled at each rung boundary and spent
                # over the next few pools, so the floor is a property of the settled
                # network for that rung and is frozen in between; a ~alpha-tail
                # quantile read off a network still chasing its data is not a
                # quantity worth acting on.  budget=None keeps the legacy behaviour.
                if self._floor_update_budget is None or self._floor_update_budget > 0:
                    pooled_w = (w_post / B).reshape(-1)                # sums to ~1
                    vals = torch.nan_to_num(log_prob_cond, nan=self.nan_replacement,
                                            neginf=self.nan_replacement).reshape(-1)
                    sorted_vals, order = torch.sort(vals)
                    cw = torch.cumsum(pooled_w[order], dim=0)
                    ge = (cw >= self.truncation_alpha).nonzero()
                    qi = int(ge[0]) if ge.numel() > 0 else sorted_vals.numel() - 1
                    q_batch = sorted_vals[qi].item()
                    if np.isfinite(q_batch):
                        m = self.truncation_momentum
                        self._logp_cond_floor = (q_batch if self._logp_cond_floor is None
                                                 else (1 - m) * self._logp_cond_floor + m * q_batch)
                    if self._floor_update_budget is not None:
                        self._floor_update_budget -= 1

                if (self._logp_cond_floor is not None
                        and self._total_epochs_trained >= self.truncation_warmup_epochs):
                    trunc = (log_prob_cond <= self._logp_cond_floor).to(log_prior.dtype) \
                            * self.out_of_bounds_penalty

        if mode == "proposal":
            if self.tight_proposal_mode:
                log_target = self.gamma * (log_prob_cond - log_prob_marg) + log_prior
            else:
                log_target = self.gamma / (1.0 + self.gamma) * log_prob_cond + log_prior
        else:  # posterior: c/m * prior
            log_target = log_prob_cond - log_prob_marg + log_prior

        log_weights = log_target - log_g_mix - mask - trunc
        log_weights = torch.nan_to_num(log_weights, nan=self.nan_replacement, neginf=self.nan_replacement)
        log_weights = log_weights - torch.logsumexp(log_weights, dim=0, keepdim=True)
        weights = torch.exp(log_weights)

        n_eff = 1.0 / (weights ** 2).sum(dim=0).cpu().numpy()
        log({"importance_sample:n_eff_min": float(n_eff.min()),
             "importance_sample:n_eff_max": float(n_eff.max())})

        if B == num_samples:
            # one column per output sample: 1 draw per column
            idx = torch.multinomial(weights.T, 1, replacement=True).squeeze(-1)
            samples_lat = proposals[idx, torch.arange(num_samples), :]
            logprob = log_prob_cond[idx, torch.arange(num_samples)].cpu()
        else:
            # Pooled SIR: num_samples draws from the single pool. Proposal draws
            # feed the simulator, so sample WITHOUT replacement there — duplicate
            # z's would be wasted simulation budget; posterior draws stay with
            # replacement (iid SIR). Fall back to replacement if the pool is
            # smaller than the request.
            replacement = mode != "proposal" or num_samples > weights.shape[0]
            idx = torch.multinomial(weights[:, 0], num_samples, replacement=replacement)
            samples_lat = proposals[idx, 0, :]
            logprob = log_prob_cond[idx, 0].cpu()
        samples = self.simulator_instance.forward(samples_lat, mode="standard_normal").cpu()
        dt = time.time() - t_start
        info(f"Time for sampling: {dt:.2f}s ({mode}, {num_samples} samples, "
             f"pool={self.num_proposals}x{B})")
        log({"importance_sample:time": dt})
        return samples, logprob.detach()

    # ==================== Save / Load ====================

    def save(self, node_dir: Path) -> None:
        node_dir = Path(node_dir)
        if not self.networks_initialized:
            raise RuntimeError("Networks not initialized.")
        torch.save(self._best_flow.state_dict(), node_dir / "flow.pth")
        torch.save(self._best_embedding.state_dict(), node_dir / "embedding.pth")
        torch.save(self._init_parameters, node_dir / "init_parameters.pth")
        torch.save(self._total_epochs_trained, node_dir / "total_epochs_trained.pth")
        torch.save(self._logp_cond_floor, node_dir / "logp_cond_floor.pth")

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

        self._best_flow.load_state_dict(torch.load(node_dir / "flow.pth"))
        self._flow.load_state_dict(self._best_flow.state_dict())
        if (node_dir / "embedding.pth").exists():
            self._best_embedding.load_state_dict(torch.load(node_dir / "embedding.pth"))
            self._embedding.load_state_dict(self._best_embedding.state_dict())

        tep = node_dir / "total_epochs_trained.pth"
        self._total_epochs_trained = torch.load(tep) if tep.exists() else 0

        fp = node_dir / "logp_cond_floor.pth"
        self._logp_cond_floor = torch.load(fp) if fp.exists() else None
