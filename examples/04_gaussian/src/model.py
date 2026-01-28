"""
Model components for the Gaussian posterior example.

This example demonstrates SNPE_gaussian with an exponential forward model:
  x = exp(z) + noise

Components:
  - ExpPlusNoise: Forward simulator (x = exp(z) + noise)
  - E_identity: Pass-through embedding (no transformation)
"""

import torch
import falcon


class ExpPlusNoise:
    """Exponential simulator: x = exp(z) + noise.

    Creates non-trivial correlations in the posterior when z has multiple dimensions.

    Args:
        sigma: Standard deviation of observation noise (default: 1e-6)
    """

    def __init__(self, sigma: float = 1e-6):
        self.sigma = sigma

    def simulate_batch(self, batch_size, z):
        z = torch.tensor(z)
        x = torch.exp(z) + torch.randn_like(z) * self.sigma
        falcon.log({"x_mean": x.mean().item()})
        falcon.log({"x_std": x.std().item()})
        return x.numpy()


class E_identity(torch.nn.Module):
    """Pass-through embedding that returns input unchanged.

    Use this when the observation x is already suitable for direct input
    to the posterior network (e.g., low-dimensional, well-scaled).
    """

    def forward(self, x):
        return x
