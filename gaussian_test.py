#!/usr/bin/env python3
"""
Minimal test for 1D Gaussian posterior learning.

Key insight: For x = z + noise, the optimal posterior is N(x, sigma^2).
The MLP must learn mean(x) ≈ x, and variance must match sigma_obs^2.

Challenge: When sigma_obs is tiny (1e-6), the mean must be extremely accurate.
"""

import argparse
import time
import torch
import torch.nn as nn
import numpy as np

def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sigma_obs', type=float, default=1e-6)
    parser.add_argument('--x_obs', type=float, default=0.3)
    parser.add_argument('--num_steps', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--print_every', type=int, default=100)
    return parser.parse_args()


class GaussianMLP(nn.Module):
    """MLP learns mean, variance is computed from residuals (closed-form optimal)."""

    def __init__(self, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        # Running estimate of variance (EMA)
        self.register_buffer('log_var', torch.tensor(0.0))
        self.var_momentum = 0.01

    def forward(self, x):
        mean = self.net(x)
        return mean, self.log_var.expand(x.shape[0], 1)

    def update_variance(self, z, x):
        """Update variance estimate from current batch (closed-form optimal)."""
        with torch.no_grad():
            mean = self.net(x)
            residuals = (z - mean) ** 2
            batch_var = residuals.mean()
            # EMA update
            self.log_var.copy_(
                (1 - self.var_momentum) * self.log_var +
                self.var_momentum * torch.log(batch_var + 1e-20)
            )

    def loss(self, z, x):
        """MSE loss for mean (variance is updated separately)."""
        mean = self.net(x)
        return ((z - mean) ** 2).mean()

    def log_prob(self, z, x):
        mean, log_var = self(x)
        var = torch.exp(log_var)
        return -0.5 * (np.log(2*np.pi) + log_var + (z - mean)**2 / var).squeeze(-1)

    def sample(self, x, n_samples=1, temperature=1.0):
        mean, log_var = self(x)
        std = torch.exp(0.5 * log_var) * np.sqrt(temperature)
        if x.shape[0] == 1:
            eps = torch.randn(n_samples, 1, device=x.device)
            return (mean + std * eps).squeeze(-1)
        return (mean + std * torch.randn_like(mean)).squeeze(-1)


def simulate(z, sigma_obs):
    return z + torch.randn_like(z) * sigma_obs

def sample_prior(n, device):
    return torch.randn(n, 1, device=device)

def sample_proposal(model, x_obs, n, gamma, device, step):
    with torch.no_grad():
        if step == 0:
            return sample_prior(n, device)
        temperature = 1.0 / gamma
        x_batch = x_obs.expand(n, -1)
        z = model.sample(x_batch, temperature=temperature)
        return z.unsqueeze(-1)


def main():
    cfg = get_config()
    torch.manual_seed(cfg.seed)
    torch.set_default_dtype(torch.float64)
    device = torch.device(cfg.device if torch.cuda.is_available() else 'cpu')

    print(f"Device: {device}")
    print(f"Problem: sigma_obs = {cfg.sigma_obs:.0e}, target log_var = {2*np.log(cfg.sigma_obs):.1f}")
    print(f"Training: {cfg.num_steps} steps × {cfg.batch_size} = {cfg.num_steps * cfg.batch_size} samples\n")

    model = GaussianMLP(hidden_dim=64).double().to(device)
    optimizer = torch.optim.Adam(model.net.parameters(), lr=cfg.lr)
    x_obs = torch.tensor([[cfg.x_obs]], device=device, dtype=torch.float64)

    print(f"{'Step':>6} {'Samples':>10} {'MSE':>12} {'Mean':>10} {'log_var':>10} {'StdRatio':>10}")
    print("-" * 70)

    t0 = time.perf_counter()
    converged_step = None

    # Warmup: learn mean=x with prior samples first
    print("[Warmup: 200 steps with prior samples]")
    for step in range(10):
        z_batch = sample_prior(cfg.batch_size, device)
        x_batch = simulate(z_batch, cfg.sigma_obs)
        optimizer.zero_grad()
        loss = model.loss(z_batch, x_batch)
        loss.backward()
        optimizer.step()
    print(f"[Warmup done, mean(x_obs)={model(x_obs)[0].item():.6f}]")

    # Lower learning rate for fine-tuning with proposal
    for pg in optimizer.param_groups:
        pg['lr'] = cfg.lr / 10
    print(f"[LR reduced to {cfg.lr/10} for proposal phase]\n")

    for step in range(cfg.num_steps):
        z_batch = sample_proposal(model, x_obs, cfg.batch_size, cfg.gamma, device, step)
        x_batch = simulate(z_batch, cfg.sigma_obs)

        # Update mean with gradient descent
        optimizer.zero_grad()
        loss = model.loss(z_batch, x_batch)
        loss.backward()
        optimizer.step()

        # Update variance with closed-form optimal
        model.update_variance(z_batch, x_batch)

        if step % cfg.print_every == 0 or step == cfg.num_steps - 1:
            mean_pred = model(x_obs)[0].item()
            log_var = model.log_var.item()
            std_pred = np.exp(0.5 * log_var)
            std_ratio = std_pred / cfg.sigma_obs
            samples = (step + 1) * cfg.batch_size

            print(f"{step:>6} {samples:>10} {loss.item():>12.2e} "
                  f"{mean_pred:>10.6f} {log_var:>10.2f} {std_ratio:>10.2f}")

            if converged_step is None and 0.9 < std_ratio < 1.1:
                converged_step = step

    elapsed = time.perf_counter() - t0
    mean_pred = model(x_obs)[0].item()
    std_pred = np.exp(0.5 * model.log_var.item())

    print()
    if converged_step is not None:
        print(f"✓ Converged at step {converged_step} ({(converged_step+1)*cfg.batch_size} samples)")

    print(f"\nFinal: mean={mean_pred:.6f} (true={cfg.x_obs:.6f}, error={abs(mean_pred-cfg.x_obs):.2e})")
    print(f"       std={std_pred:.2e} (true={cfg.sigma_obs:.2e})")
    print(f"       std_ratio={std_pred/cfg.sigma_obs:.3f}")
    print(f"\nTime: {elapsed:.2f}s")


if __name__ == "__main__":
    main()
