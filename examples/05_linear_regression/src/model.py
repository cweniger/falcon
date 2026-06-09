"""
Model components for the linear regression example.

Linear model: x = Phi @ theta + noise
  - Phi[i, k] = sin((k+1) * t_i), t_i on [0, 2*pi], 1000 bins, 10 parameters
  - noise ~ N(0, sigma^2 * I)

Components:
  - SignalSimulator: Noiseless forward model (mu = Phi @ theta)
  - NoiseSimulator:  Adds Gaussian noise (x = mu + noise)
  - LinearSimulator: Combined simulator (kept for extras/standalone.py)
"""

import torch
import torch.nn as nn
import numpy as np
import falcon


def design_matrix(n_bins=1000, n_params=10):
    """Build design matrix Phi[i, k] = sin((k+1) * t_i)."""
    t = np.linspace(0, 2 * np.pi, n_bins, endpoint=False)
    Phi = np.column_stack([np.sin((k + 1) * t) for k in range(n_params)])
    return Phi, t


class SignalSimulator:
    """Noiseless forward model: mu = Phi @ theta."""

    def __init__(self, n_bins: int = 1000, n_params: int = 10):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        Phi, _ = design_matrix(n_bins, n_params)
        self.Phi = torch.tensor(Phi, device=self.device)

    def simulate_batch(self, batch_size, theta):
        theta = torch.tensor(theta, device=self.device)
        Phi = self.Phi.to(dtype=theta.dtype)
        return (theta @ Phi.T).cpu().numpy()


class NoiseSimulator:
    """Adds Gaussian noise: x = mu + noise."""

    def __init__(self, sigma: float = 0.1):
        self.sigma = sigma
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def simulate_batch(self, batch_size, mu):
        mu = torch.tensor(mu, device=self.device)
        noise = torch.randn_like(mu) * self.sigma
        x = mu + noise
        falcon.log({"x_mean": x.mean().item(), "x_std": x.std().item()})
        return x.cpu().numpy()


class LinearSimulator:
    """Combined simulator: x = Phi @ theta + noise (kept for extras/standalone.py)."""

    def __init__(self, sigma: float = 0.1, n_bins: int = 1000, n_params: int = 10):
        self.sigma = sigma
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        Phi, _ = design_matrix(n_bins, n_params)
        self.Phi = torch.tensor(Phi, device=self.device)

    def simulate_batch(self, batch_size, theta):
        theta = torch.tensor(theta, device=self.device)
        Phi = self.Phi.to(dtype=theta.dtype)
        noise = torch.randn(batch_size, Phi.shape[0], device=self.device, dtype=theta.dtype) * self.sigma
        x = theta @ Phi.T + noise
        falcon.log({"x_mean": x.mean().item(), "x_std": x.std().item()})
        return x.cpu().numpy()


class E_fft_norm(nn.Module):
    """FFT-based embedding with gated linear projection."""

    def __init__(self, n_bins: int = 1000, n_features: int = 128, n_modes: int = 128):
        super().__init__()
        self.n_modes = n_modes
        self.linear = nn.Linear(n_modes * 2, n_features)
        self.gate = nn.Linear(n_modes * 2, n_features)

    def forward(self, x):
        x = x.float()
        f = torch.fft.rfft(x, norm='ortho')[..., :self.n_modes]
        h = torch.cat([f.real, f.imag], dim=-1)
        return self.linear(h) * torch.sigmoid(self.gate(h))


class E_fft_whiten(nn.Module):
    """FFT embedding with diagonal whitening (EMA-updated) on raw input."""

    def __init__(self, n_bins: int = 1000, n_features: int = 128, n_modes: int = 128,
                 momentum: float = 0.01, min_var: float = 1e-20):
        super().__init__()
        self.n_modes = n_modes
        self.momentum = momentum
        self.min_var = min_var
        self.register_buffer('_input_mean', torch.zeros(n_bins))
        self.register_buffer('_input_std', torch.ones(n_bins))
        self.linear = nn.Linear(n_modes * 2, n_features)

    def forward(self, x):
        x = x.float()
        if self.training:
            with torch.no_grad():
                m = self.momentum
                self._input_mean = (1 - m) * self._input_mean + m * x.mean(dim=0)
                self._input_std = (1 - m) * self._input_std + m * x.var(dim=0).clamp(min=self.min_var).sqrt()
        x_white = (x - self._input_mean.detach()) / self._input_std.detach()
        f = torch.fft.rfft(x_white, norm='ortho')[..., :self.n_modes]
        h = torch.cat([f.real, f.imag], dim=-1)
        return self.linear(h)


class E_linear(nn.Module):
    """Linear embedding: projects data down to a lower-dimensional summary."""

    def __init__(self, n_bins: int = 1000, n_out: int = 20):
        super().__init__()
        self.linear = nn.Linear(n_bins, n_out)

    def forward(self, x):
        return self.linear(x.float())
