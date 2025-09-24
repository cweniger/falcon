# -*- coding: utf-8 -*-
"""
Bimodal 10D example implemented with the same class/function names as 04_bimodal.
Generation law matches theoretical code:
    x = z + N(0, sigma^2 I) + mode_shift,  mode_shift_d ∈ { -3σ, +3σ } independently per dim.
"""

import torch
import falcon
import numpy as np

# import wandb

DIM = 10
MU = 0.0
SIGMA = 1e-2  # 0.01
MEAN_SHIFT = 3.0 * SIGMA
PRIOR_SCALE = 1000_000
# calculate prior range
PRIOR_LOW = MU - PRIOR_SCALE * SIGMA
PRIOR_HIGH = MU + PRIOR_SCALE * SIGMA

NPAR = DIM  # parameter dimetions == 10
NBINS = DIM  # Observation dimension == 10 (key: let x be a 10-dimensional vector)


class Signal:
    def simulate_batch(self, batch_size, z):
        z = torch.as_tensor(z, dtype=torch.float64)  # (B, DIM)
        assert z.ndim == 2 and z.shape[1] == DIM, f"z shape={z.shape} != (B,{DIM})"
        # bits = torch.randint(0, 2, (batch_size, DIM), dtype=torch.int64)     # 0/1
        # shift = torch.where(bits == 0,
        #                     torch.full_like(z, -MEAN_SHIFT),
        #                     torch.full_like(z,  MEAN_SHIFT))                 # (B, DIM)
        # bits = torch.randint(0, 2, (batch_size, 1), dtype=torch.int64, device=z.device)
        # shift = torch.where(bits == 0, -MEAN_SHIFT, MEAN_SHIFT).to(z.dtype)  # (B,1)
        self.mean_shift = MEAN_SHIFT  # distance between the two modes
        self.modes = [
            torch.full((DIM,), MU - self.mean_shift, dtype=torch.float64),  #  (DIM,)
            torch.full((DIM,), MU + self.mean_shift, dtype=torch.float64),  #  (DIM,)
        ]
        # choose mode based on a random binary vector
        mode_choice = torch.randint(0, 2, (batch_size, DIM), dtype=torch.float64)
        mode_shift = (1 - mode_choice) * self.modes[0] + mode_choice * self.modes[1]
        m = z + mode_shift  # (B, DIM)
        falcon.log({"Signal:mean": m.mean().item()})
        falcon.log({"Signal:std": m.std().item()})
        return m.numpy()


class Noise:
    """Gaussian noise generator."""

    def simulate(self):
        result = torch.randn((NBINS,)).double() * SIGMA
        falcon.log({"Noise:mean": result.mean().item()})
        falcon.log({"Noise:std": result.std().item()})
        return result.numpy()


class Add:
    """Adds signal and noise components."""

    def simulate_batch(self, batch_size, m, n):
        m = torch.tensor(m)  # Input is numpy array
        n = torch.tensor(n)  # Input is numpy array
        result = m + n
        falcon.log({"Add:mean": result.mean().item()})
        falcon.log({"Add:std": result.std().item()})
        return result.numpy()


# class E(torch.nn.Module):
#     """Embedding network with PCA projection and whitening.

#     This network implements streaming PCA and whitening for dimensionality
#     reduction and preprocessing of high-dimensional observations.
#     """
#     def __init__(self, log_prefix=None):
#         super(E, self).__init__()
#         # Import PCA components from their specific modules
#         from falcon.contrib.svd import PCAProjector
#         from falcon.contrib.norms import DiagonalWhitener

#         self.projector = PCAProjector(buffer_size=128)
#         self.whitener = DiagonalWhitener(NBINS, use_fourier=False)
#         self.linear = torch.nn.LazyLinear(NPAR * 2)
#         self.log_prefix = log_prefix + ":" if log_prefix else ""

#     def forward(self, x, *args):
#         falcon.log({f"{self.log_prefix}input_min": x.min().item()})
#         falcon.log({f"{self.log_prefix}input_max": x.max().item()})
#         if args[0] is not None and self.training:  # Scaffolds provided
#             m, n = args
#             # Update whitener with noise component
#             self.whitener.update(n+m)
#             # Apply whitening to signal component
#             white_m = self.whitener(m)
#             # Update PCA projector with whitened signal
#             self.projector.update(white_m)

#         # Apply PCA projection to observation
#         try:
#             #falcon.log({"E:x_input_histogram": wandb.Histogram(x.detach().cpu().flatten().numpy())})
#             x = self.whitener(x)
#             #falcon.log({"E:x_whitened_histogram": wandb.Histogram(x.detach().cpu().flatten().numpy())})
#             x = self.projector(x).float()
#             #falcon.log({"E:x_projected_histogram": wandb.Histogram(x.detach().cpu().flatten().numpy())})
#         except:
#             x = x.float()

#         # Apply linear transformation
#         x = x/NBINS
#         x = self.linear(x)
#         #falcon.log({"E:x_linear_compression": wandb.Histogram(x.detach().cpu().flatten().numpy())})


#         falcon.log({f"{self.log_prefix}output_min": x.min().item()})
#         falcon.log({f"{self.log_prefix}output_max": x.max().item()})
#         return x
class E(torch.nn.Module):
    def __init__(self, log_prefix=None):
        super().__init__()
        from falcon.contrib.norms import LazyOnlineNorm

        self.norm = LazyOnlineNorm(momentum=5e-3)
        self.linear = torch.nn.Linear(DIM, DIM * 2)
        self.log_prefix = (log_prefix + ":") if log_prefix else ""

    def forward(self, x, *args):
        x = self.norm(x).float()
        y = self.linear(x)
        falcon.log({f"{self.log_prefix}embed_min": y.min().item()})
        falcon.log({f"{self.log_prefix}embed_max": y.max().item()})
        return y
