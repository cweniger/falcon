"""Conditional flow-matching backend (velocity field, Euler sampling, CNF density).

Framework-agnostic building blocks used by the GaussianizedFlowMatching estimator.
All functions operate on flat 2-D tensors:

    w : (M, param_dim)      points in the (whitened) target space
    s : (M, cond_dim)       per-point conditioning embedding (zeros for marginal)
    t : (M, 1) or scalar    time in [0, 1]

The model is a velocity field ``v(w, t, s)`` trained by flow matching with the
straight-line interpolant (base N(0,I) at t=0 -> data at t=1). Sampling integrates
the ODE forward with Euler; density uses the continuous change-of-variables with the
ODE run backward, accumulating the divergence (exact trace or Hutchinson estimate).
"""

from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn


class GaussianFourierTime(nn.Module):
    """Random Fourier features for the scalar time input."""

    def __init__(self, dim: int, scale: float = 3.0):
        super().__init__()
        assert dim % 2 == 0, "time_dim must be even"
        self.register_buffer("freqs", torch.randn(dim // 2) * scale)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        proj = 2 * math.pi * t * self.freqs
        return torch.cat([proj.sin(), proj.cos()], dim=-1)


class VelocityField(nn.Module):
    """MLP velocity field v(w, t, s) for conditional flow matching."""

    def __init__(self, param_dim: int, cond_dim: int, hidden: int = 256,
                 layers: int = 4, time_dim: int = 64, layernorm: bool = True):
        super().__init__()
        self.param_dim = param_dim
        self.cond_dim = cond_dim
        self.time_embed = GaussianFourierTime(time_dim)
        dims = [param_dim + time_dim + cond_dim] + [hidden] * layers
        net = []
        for d_in, d_out in zip(dims[:-1], dims[1:]):
            net += [nn.Linear(d_in, d_out)]
            if layernorm:                       # hidden layers only (never the input cat or output)
                net += [nn.LayerNorm(d_out)]
            net += [nn.SiLU()]
        net += [nn.Linear(hidden, param_dim)]
        self.net = nn.Sequential(*net)

    def forward(self, w: torch.Tensor, t: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        if t.ndim == 1:
            t = t[:, None]
        return self.net(torch.cat([w, self.time_embed(t), s], dim=-1))


class EMA:
    """Maintains an exponential moving average copy of a velocity field."""

    def __init__(self, decay: float = 0.999):
        self.decay = decay

    @torch.no_grad()
    def update(self, ema_model: nn.Module, model: nn.Module) -> None:
        d = self.decay
        for pe, p in zip(ema_model.parameters(), model.parameters()):
            pe.mul_(d).add_(p.detach(), alpha=1 - d)
        for be, b in zip(ema_model.buffers(), model.buffers()):
            be.copy_(b)

    @staticmethod
    def clone(model: nn.Module) -> nn.Module:
        ema = copy.deepcopy(model)
        for p in ema.parameters():
            p.requires_grad_(False)
        return ema


# --------------------------------------------------------------------------- #
# Training loss
# --------------------------------------------------------------------------- #
def fm_loss(net: VelocityField, w1: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """Flow-matching loss ‖v(w_t, t, s) − (w1 − w0)‖² with w0~N(0,I), t~U(0,1)."""
    w0 = torch.randn_like(w1)
    t = torch.rand(w1.shape[0], 1, device=w1.device, dtype=w1.dtype)
    wt = (1 - t) * w0 + t * w1
    target = w1 - w0
    return (net(wt, t, s) - target).pow(2).mean()


# --------------------------------------------------------------------------- #
# Sampling: forward Euler, t: 0 -> 1
# --------------------------------------------------------------------------- #
@torch.no_grad()
def euler_sample(net: VelocityField, s: torch.Tensor, param_dim: int, steps: int) -> torch.Tensor:
    """Draw w ~ model conditioned on s (one sample per row of s)."""
    m = s.shape[0]
    w = torch.randn(m, param_dim, device=s.device, dtype=s.dtype)
    dt = 1.0 / steps
    for i in range(steps):
        tb = torch.full((m, 1), i * dt, device=s.device, dtype=w.dtype)
        w = w + dt * net(w, tb, s)
    if w.abs().max().item() > 3:
        info("Large Euler sample values: "+str(w))
    return w


# --------------------------------------------------------------------------- #
# Velocity + divergence tr(∂v/∂w) for the CNF density
# --------------------------------------------------------------------------- #
def _vel_and_div(net, w, t, s, divergence, n_probe):
    """Velocity v(w,t,s) and tr(∂v/∂w): exact (param_dim VJPs) or Hutchinson (n_probe probes)."""
    with torch.enable_grad():
        w = w.detach().requires_grad_(True)
        v = net(w, t, s)
        div = torch.zeros(w.shape[0], device=w.device, dtype=w.dtype)
        if divergence == "exact":
            for i in range(w.shape[1]):
                gi = torch.autograd.grad(v[:, i].sum(), w, retain_graph=(i < w.shape[1] - 1))[0]
                div = div + gi[:, i]
        else:
            for _ in range(n_probe):
                eps = torch.randint(0, 2, w.shape, device=w.device, dtype=w.dtype) * 2 - 1
                g = torch.autograd.grad(v, w, grad_outputs=eps, retain_graph=True)[0]
                div = div + (g * eps).sum(1)
            div = div / n_probe
    return v.detach(), div.detach()


# --------------------------------------------------------------------------- #
# Density: backward Euler CNF, t: 1 -> 0
# --------------------------------------------------------------------------- #
def cnf_logprob(net: VelocityField, w: torch.Tensor, s: torch.Tensor, steps: int,
                divergence: str = "hutch", n_probe: int = 4) -> torch.Tensor:
    """log density of the model at points w (conditioned on s), in the model's space.

    log p(w) = log N(z0; 0, I) + INT_{t=1}^{0} (∇·v) dt   (Euler integration).
    """
    z = w
    logdet = torch.zeros(w.shape[0], device=w.device, dtype=w.dtype)
    dt = -1.0 / steps
    for i in range(steps):
        tb = torch.full((z.shape[0], 1), 1.0 + i * dt, device=z.device, dtype=z.dtype)
        v, d = _vel_and_div(net, z, tb, s, divergence, n_probe)
        z = z + dt * v
        logdet = logdet + dt * d
    base = -0.5 * (z.pow(2).sum(1) + z.shape[1] * math.log(2 * math.pi))
    return base + logdet
