#!/usr/bin/env python3
"""
Demonstration of explosion mechanism in sequential NPE with Gaussian posteriors.

Key insight: When |bias| = |mean_pred - x_obs| exceeds σ_obs, the system transitions
from stable "identity regime" (batch_var ≈ σ_obs²) to unstable "constant regime"
(batch_var ≈ bias²), causing exponential divergence.

See EXPLOSION_NOTES.md for full explanation.
"""

import torch
import torch.nn as nn
import numpy as np

torch.set_default_dtype(torch.float64)


class GaussianMLP(nn.Module):
    """MLP predicting mean, with variance estimated via EMA of residuals."""

    def __init__(self, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        self.register_buffer('log_var', torch.tensor(0.0))
        self.var_momentum = 0.01

    def loss(self, z, x):
        return ((z - self.net(x)) ** 2).mean()

    def update_variance(self, z, x):
        with torch.no_grad():
            batch_var = ((z - self.net(x)) ** 2).mean()
            self.log_var.copy_((1 - self.var_momentum) * self.log_var +
                               self.var_momentum * torch.log(batch_var + 1e-40))


def run_demonstration():
    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Problem setup
    sigma_obs = 1e-6
    x_obs_val = 0.3
    batch_size = 128
    temperature = 2.0

    model = GaussianMLP(hidden_dim=64).double().to(device)
    optimizer = torch.optim.Adam(model.net.parameters(), lr=0.01)
    x_obs = torch.tensor([[x_obs_val]], device=device, dtype=torch.float64)

    # Brief warmup with prior samples
    for _ in range(10):
        z = torch.randn(batch_size, 1, device=device)
        x = z + torch.randn_like(z) * sigma_obs
        optimizer.zero_grad()
        model.loss(z, x).backward()
        optimizer.step()

    for pg in optimizer.param_groups:
        pg['lr'] = 0.001

    print("Explosion Mechanism Demonstration")
    print("=" * 95)
    print()
    print("bias = mean_pred - x_obs")
    print("Critical threshold: |bias| ≈ σ_obs")
    print("  - |bias| << σ_obs: batch_var ≈ σ_obs² (stable)")
    print("  - |bias| >> σ_obs: batch_var ≈ bias² (unstable → explosion)")
    print()
    print(f"{'Step':>5} | {'bias':>12} {'|bias|/σ':>10} | {'batch_var':>12} {'σ_obs²':>10} {'bias²':>12} | {'std_ratio':>10}")
    print("-" * 95)

    for step in range(6000):
        with torch.no_grad():
            mean_pred = model.net(x_obs)
            std_pred = torch.exp(0.5 * model.log_var) * np.sqrt(temperature)
            z = mean_pred + std_pred * torch.randn(batch_size, 1, device=device)

        x = z + torch.randn_like(z) * sigma_obs

        with torch.no_grad():
            batch_var = ((z - model.net(x)) ** 2).mean().item()

        optimizer.zero_grad()
        loss = model.loss(z, x)
        loss.backward()
        optimizer.step()
        model.update_variance(z, x)

        if step % 500 == 0 or (step >= 4900 and step % 100 == 0):
            with torch.no_grad():
                std_ratio = np.exp(0.5 * model.log_var.item()) / sigma_obs
                mean_pred_val = mean_pred.item()
                bias = mean_pred_val - x_obs_val
                bias_ratio = abs(bias) / sigma_obs

            # Mark the transition
            marker = " ← threshold!" if 0.8 < bias_ratio < 1.2 else ""
            marker = " ← EXPLOSION" if bias_ratio > 100 else marker

            print(f"{step:>5} | {bias:>+12.2e} {bias_ratio:>10.1f} | "
                  f"{batch_var:>12.2e} {sigma_obs**2:>10.2e} {bias**2:>12.2e} | "
                  f"{std_ratio:>10.2f}{marker}")

    print()
    print("Summary:")
    print("  - Steps 0-5000: |bias|/σ < 1, batch_var ≈ σ_obs², stable at std_ratio ≈ 1")
    print("  - Step ~5000: |bias|/σ crosses 1 (threshold)")
    print("  - Steps 5000+: |bias|/σ >> 1, batch_var ≈ bias², explosion")


if __name__ == "__main__":
    run_demonstration()
