"""
Model components for the linear regression example.

Linear model: y = Phi @ theta + noise
  - Phi[i, k] = sin((k+1) * x_i), x_i on [0, 2*pi], 20000 bins, 10 parameters
  - noise ~ N(0, sigma^2 * I)

Components:
  - LinearSimulator: Forward simulator (y = Phi @ theta + noise)
  - E_fft_norm: FFT-based embedding with gated linear projection
  - E_linear: Linear embedding network (kept for backwards compatibility)
"""

import torch
import torch.nn as nn
import numpy as np
import falcon


def design_matrix(n_bins=20000, n_params=10):
    """Build design matrix Phi[i, k] = sin((k+1) * x_i)."""
    x = np.linspace(0, 2 * np.pi, n_bins, endpoint=False)
    Phi = np.column_stack([np.sin((k + 1) * x) for k in range(n_params)])
    return Phi, x


class LinearSimulator:
    """Linear regression simulator: y = Phi @ theta + noise.

    Args:
        sigma: Standard deviation of observation noise per bin (default: 0.1)
        n_bins: Number of data bins (default: 100)
        n_params: Number of parameters (default: 10)
    """

    def __init__(self, sigma: float = 0.1, n_bins: int = 20000, n_params: int = 10):
        self.sigma = sigma
        Phi, _ = design_matrix(n_bins, n_params)
        self.Phi = torch.tensor(Phi)

    def simulate_batch(self, batch_size, theta):
        theta = torch.tensor(theta)
        Phi = self.Phi.to(dtype=theta.dtype)
        y = theta @ Phi.T + torch.randn(batch_size, Phi.shape[0], dtype=theta.dtype) * self.sigma
        falcon.log({"y_mean": y.mean().item(), "y_std": y.std().item()})
        return y.numpy()


class E_fft_norm(nn.Module):
    """FFT-based embedding with gated linear projection.

    Computes the orthonormal FFT of the input, truncates to n_modes,
    stacks real and imaginary parts, and applies a gated linear layer.

    Args:
        n_bins: Number of input bins (default: 20000)
        n_features: Output embedding dimension (default: 128)
        n_modes: Number of FFT modes to keep (default: 128)
    """

    def __init__(self, n_bins: int = 20000, n_features: int = 128, n_modes: int = 128):
        super().__init__()
        self.n_modes = n_modes
        self.linear = nn.Linear(n_modes * 2, n_features)
        self.gate = nn.Linear(n_modes * 2, n_features)

    def forward(self, x):
        # Orthonormal FFT, truncate to n_modes
        f = torch.fft.rfft(x, norm='ortho')[..., :self.n_modes]
        # Stack real and imaginary parts
        h = torch.cat([f.real, f.imag], dim=-1)
        return self.linear(h) * torch.sigmoid(self.gate(h))


class E_linear(nn.Module):
    """Linear embedding: projects data down to a lower-dimensional summary.

    Args:
        n_bins: Number of input bins (default: 20000)
        n_out: Output embedding dimension (default: 20)
    """

    def __init__(self, n_bins: int = 20000, n_out: int = 20):
        super().__init__()
        self.linear = nn.Linear(n_bins, n_out)

    def forward(self, x):
        return self.linear(x)
