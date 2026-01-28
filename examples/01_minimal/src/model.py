"""
Example model implementation for 01_minimal.
This file contains model-specific components that can be referenced via _target_
"""

import torch
import falcon
import numpy as np

# Global configuration
SIGMA = 0.1  # Hardcoded sigma value


class Simulate:
    """Gaussian simulator for generating observations.

    Args:
        npar: Number of parameters (required)
        sigma: Standard deviation for noise (default: 0.1)
    """

    def __init__(self, npar: int, sigma: float = 1e-1):
        self.npar = npar
        self.sigma = sigma

    def simulate_batch(self, batch_size, z):
        z = torch.tensor(z)  # Input is numpy array
        x = z + torch.randn_like(z) * self.sigma
        falcon.log({"x_mean": x.mean().item()})
        falcon.log({"x_std": x.std().item()})
        return x.numpy()  # Return numpy array


class E(torch.nn.Module):
    """Embedding network with online normalization.

    Args:
        momentum: Momentum for online normalization (default: 0.01)
    """

    def __init__(self, momentum: float = 1e-2):
        super(E, self).__init__()
        self.norm = falcon.contrib.LazyOnlineNorm(momentum=momentum)

    def forward(self, x):
        falcon.log({"norm_pre": x.std().item()})
        x = self.norm(x).float()
        falcon.log({"norm_post": x.std().item()})
        return x
