"""
Example model implementation for adaptive.py
This file contains model-specific components that can be referenced via _target_
"""

import torch
import falcon

# Global configuration
SIGMA = 0.1  # Hardcoded sigma value


class Simulate:
    """Gaussian simulator compatible with the newer API.

    Implements `simulate_batch(batch_size, z)` and `get_shape_and_dtype()`.
    """
    def __init__(self, npar: int = 3, sigma: float = 1e-1):
        self.npar = npar
        self.sigma = sigma

    def simulate_batch(self, batch_size, z):
        # z expected as array-like with shape (batch_size, npar)
        z = torch.tensor(z)
        noise = torch.randn_like(z) * self.sigma
        x = z + noise
        falcon.log({"Simulate:mean": x.mean().item()})
        falcon.log({"Simulate:std": x.std().item()})
        return x.numpy()

    def get_shape_and_dtype(self):
        return (self.npar,), 'float64'


class E(torch.nn.Module):
    """Embedding network with online normalization.
    
    Args:
        momentum: Momentum for online normalization (default: 0.01)
    """
    def __init__(self, momentum: float = 1e-2):
        super(E, self).__init__()
        # Keep embedding minimal: use online norm if available, otherwise a simple layer
        self.momentum = momentum
        # Try to use provided contrib normalization if present
        try:
            self.norm = falcon.contrib.LazyOnlineNorm(momentum=momentum)
        except Exception:
            self.norm = None
        self.linear = torch.nn.LazyLinear(8)

    def forward(self, x):
        falcon.log({"E:input_min": x.min().item()})
        falcon.log({"E:input_max": x.max().item()})
        if self.norm is not None:
            x = self.norm(x).float()
        else:
            x = (x - x.mean()) / (x.std() + 1e-6)

        x = self.linear(x.float())
        falcon.log({"E:output_min": x.min().item()})
        falcon.log({"E:output_max": x.max().item()})
        return x
