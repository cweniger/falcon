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
        device: Auto-detected (uses CUDA if available). GPU accelerates the
            large matrix multiply for high n_bins.
    """

    def __init__(self, sigma: float = 0.1, n_bins: int = 20000, n_params: int = 10):
        self.sigma = sigma
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        Phi, _ = design_matrix(n_bins, n_params)
        self.Phi = torch.tensor(Phi, device=self.device)

    def simulate_batch(self, batch_size, theta):
        theta = torch.tensor(theta, device=self.device)
        Phi = self.Phi.to(dtype=theta.dtype)
        y = theta @ Phi.T + torch.randn(batch_size, Phi.shape[0], device=self.device, dtype=theta.dtype) * self.sigma
        falcon.log({"y_mean": y.mean().item(), "y_std": y.std().item()})
        return y.cpu().numpy()


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


class E_fft_whiten(nn.Module):
    """FFT embedding with built-in diagonal whitening on raw input.

    Matches the standalone.py pipeline: whiten(raw obs) -> ortho FFT -> linear.
    This ensures the FFT operates on centered, standardized data.

    The whitening statistics are updated via EMA during training.

    Args:
        n_bins: Number of input bins (default: 20000)
        n_features: Output embedding dimension (default: 128)
        n_modes: Number of FFT modes to keep (default: 128)
        momentum: EMA momentum for whitening statistics (default: 0.01)
        min_var: Minimum variance for numerical stability (default: 1e-20)
    """

    def __init__(self, n_bins: int = 20000, n_features: int = 128, n_modes: int = 128,
                 momentum: float = 0.01, min_var: float = 1e-20):
        super().__init__()
        self.n_modes = n_modes
        self.momentum = momentum
        self.min_var = min_var

        # Whitening statistics on raw input
        self.register_buffer('_input_mean', torch.zeros(n_bins))
        self.register_buffer('_input_std', torch.ones(n_bins))

        # Single linear projection (matches standalone FFTNormEmbedding)
        self.linear = nn.Linear(n_modes * 2, n_features)

    def forward(self, x):
        # Update whitening stats during training
        if self.training:
            with torch.no_grad():
                self._input_mean.lerp_(x.mean(dim=0), self.momentum)
                batch_var = x.var(dim=0).clamp(min=self.min_var)
                self._input_std.lerp_(batch_var.sqrt(), self.momentum)

        # Whiten, then FFT
        x_white = (x - self._input_mean.detach()) / self._input_std.detach()
        f = torch.fft.rfft(x_white, norm='ortho')[..., :self.n_modes]
        h = torch.cat([f.real, f.imag], dim=-1)
        return self.linear(h)


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
