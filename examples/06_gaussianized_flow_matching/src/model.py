"""Moderate 5D bimodal example for GaussianizedFlowMatching.

x = z + shift + N(0, sigma^2),  shift_d in {-SHIFT, +SHIFT} independently per dim.
Prior z ~ U(-3, 3)^5. Modes are well-separated from the prior scale (SHIFT/range ~ 0.08),
so the Gaussian preconditioner + flow-matching residual can resolve the bimodal structure.
Posterior p(z | x_obs) has 2^5 modes at  z_d = x_obs_d -/+ SHIFT,  width sigma.
"""
import torch
import numpy as np
import falcon

DIM = 5
SIGMA = 0.001
SHIFT = 0.005   # unimodal: posterior z|x ~ N(x, ~SIGMA^2)


class Signal:
    def simulate_batch(self, batch_size, z):
        z = torch.as_tensor(z, dtype=torch.float64)
        mode_choice = torch.randint(0, 2, (batch_size, DIM), dtype=torch.float64)
        shift = (2 * mode_choice - 1) * SHIFT
        m = z + shift
        falcon.log({"Signal:std": m.std().item()})
        return m.numpy()


class Noise:
    def simulate(self):
        return (torch.randn(DIM).double() * SIGMA).numpy()


class Add:
    def simulate_batch(self, batch_size, m, n):
        return (torch.tensor(m) + torch.tensor(n)).numpy()


class E(torch.nn.Module):
    def __init__(self, log_prefix=None):
        super().__init__()
        from falcon.embeddings import LazyOnlineNorm
        self.norm = LazyOnlineNorm(momentum=1e-2)
        self.linear = torch.nn.Linear(DIM, DIM * 2)

    def forward(self, x, *args):
        return self.linear(self.norm(x).float())
