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
    parser.add_argument('--num_warmup', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--print_every', type=int, default=100)
    parser.add_argument('--iw', action='store_true', help='Enable importance weighting (prior/proposal)')
    parser.add_argument('--prior_fraction', type=float, default=0.0,
                        help='Fraction of batch from prior (rest from proposal). Maintains global structure.')
    parser.add_argument('--prop_freeze', type=int, default=0,
                        help='Freeze proposal after this many steps (0 = never freeze)')
    parser.add_argument('--normalize', action='store_true',
                        help='Use normalized MLP with online input/output normalization (stable!)')
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

    def update_stats(self, z, x):
        """Update running statistics (for compatibility with NormalizedMLP)."""
        pass  # No-op for standard MLP

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

    def loss(self, z, x, weights=None):
        """MSE loss for mean (variance is updated separately)."""
        mean = self.net(x)
        squared_errors = (z - mean) ** 2
        if weights is not None:
            return (squared_errors.squeeze(-1) * weights).sum()
        return squared_errors.mean()

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


class NormalizedMLP(nn.Module):
    """MLP with online input/output normalization for stability.

    Prediction: mu(x) = output_mean + output_std * mlp((x - input_mean) / input_std)

    The running stats track the data distribution, so the MLP always operates
    in a normalized space (~N(0,1) inputs and outputs), making it stable even
    when training on extremely narrow distributions.
    """

    def __init__(self, hidden_dim=64, momentum=0.01):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        # Input normalization (running stats)
        self.register_buffer('input_mean', torch.tensor(0.0))
        self.register_buffer('input_std', torch.tensor(1.0))
        # Output denormalization (running stats)
        self.register_buffer('output_mean', torch.tensor(0.0))
        self.register_buffer('output_std', torch.tensor(1.0))
        # Variance estimate
        self.register_buffer('log_var', torch.tensor(0.0))
        self.var_momentum = momentum

    def update_stats(self, z, x):
        """Update running input/output statistics."""
        with torch.no_grad():
            # Input stats
            batch_mean_x = x.mean()
            batch_std_x = x.std() + 1e-8
            self.input_mean.copy_((1 - self.var_momentum) * self.input_mean + self.var_momentum * batch_mean_x)
            self.input_std.copy_((1 - self.var_momentum) * self.input_std + self.var_momentum * batch_std_x)
            # Output stats
            batch_mean_z = z.mean()
            batch_std_z = z.std() + 1e-8
            self.output_mean.copy_((1 - self.var_momentum) * self.output_mean + self.var_momentum * batch_mean_z)
            self.output_std.copy_((1 - self.var_momentum) * self.output_std + self.var_momentum * batch_std_z)

    def forward(self, x):
        # Normalize input
        x_norm = (x - self.input_mean) / (self.input_std + 1e-8)
        # MLP in normalized space
        r = self.net(x_norm)
        # Denormalize output
        mean = self.output_mean + self.output_std * r
        return mean, self.log_var.expand(x.shape[0], 1)

    def update_variance(self, z, x):
        """Update variance estimate from residuals."""
        with torch.no_grad():
            mean, _ = self(x)
            residuals = (z - mean) ** 2
            batch_var = residuals.mean()
            self.log_var.copy_(
                (1 - self.var_momentum) * self.log_var +
                self.var_momentum * torch.log(batch_var + 1e-20)
            )

    def loss(self, z, x, weights=None):
        """MSE loss for mean."""
        mean, _ = self(x)
        squared_errors = (z - mean) ** 2
        if weights is not None:
            return (squared_errors.squeeze(-1) * weights).sum()
        return squared_errors.mean()

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

def sample_proposal(model, x_obs, n, gamma, device, step, frozen_params=None):
    with torch.no_grad():
        if step == 0:
            return sample_prior(n, device)
        temperature = 1.0 / gamma

        if frozen_params is not None:
            # Use frozen proposal parameters
            mean, std = frozen_params
            z = mean + std * torch.randn(n, device=device)
            return z.unsqueeze(-1)

        # Use current model predictions
        x_batch = x_obs.expand(n, -1)
        z = model.sample(x_batch, temperature=temperature)
        return z.unsqueeze(-1)


def get_proposal_params(model, x_obs, gamma):
    """Get current proposal parameters (mean, std) for freezing."""
    with torch.no_grad():
        mean_pred, log_var = model(x_obs)
        temperature = 1.0 / gamma
        std = torch.exp(0.5 * log_var) * np.sqrt(temperature)
        return mean_pred.item(), std.item()

def compute_importance_weights(z, model, x_obs, gamma):
    """Compute normalized importance weights: prior(z) / proposal(z).

    Prior: N(0, 1)
    Proposal: N(mean_pred, var_learned * temperature)

    Returns normalized weights that sum to 1.
    """
    with torch.no_grad():
        z_flat = z.squeeze(-1)
        temperature = 1.0 / gamma

        # Prior log prob: N(0, 1)
        log_prior = -0.5 * z_flat ** 2

        # Proposal log prob: N(mean_pred, var * temperature)
        mean_pred, log_var = model(x_obs)
        mean_pred = mean_pred.squeeze()
        var_proposal = torch.exp(log_var.squeeze()) * temperature
        log_proposal = -0.5 * (z_flat - mean_pred) ** 2 / var_proposal - 0.5 * torch.log(var_proposal)

        # Log importance weights
        log_weights = log_prior - log_proposal

        # Normalize weights (softmax for numerical stability)
        log_weights = log_weights - log_weights.max()
        weights = torch.exp(log_weights)
        weights = weights / weights.sum()

        return weights


def main():
    cfg = get_config()
    torch.manual_seed(cfg.seed)
    torch.set_default_dtype(torch.float64)
    device = torch.device(cfg.device if torch.cuda.is_available() else 'cpu')

    print(f"Device: {device}")
    print(f"Problem: sigma_obs = {cfg.sigma_obs:.0e}, target log_var = {2*np.log(cfg.sigma_obs):.1f}")
    print(f"Training: {cfg.num_steps} steps × {cfg.batch_size} = {cfg.num_steps * cfg.batch_size} samples")
    if cfg.iw:
        print(f"Importance weighting: ENABLED (reweighting proposal → prior)")
    if cfg.prior_fraction > 0:
        print(f"Mixed sampling: {cfg.prior_fraction*100:.0f}% prior + {(1-cfg.prior_fraction)*100:.0f}% proposal")
    if cfg.prop_freeze > 0:
        print(f"Proposal freeze: after step {cfg.prop_freeze}")
    if cfg.normalize:
        print(f"Using NormalizedMLP (online input/output normalization)")
    print()

    if cfg.normalize:
        model = NormalizedMLP(hidden_dim=64).double().to(device)
    else:
        model = GaussianMLP(hidden_dim=64).double().to(device)
    optimizer = torch.optim.Adam(model.net.parameters(), lr=cfg.lr)
    x_obs = torch.tensor([[cfg.x_obs]], device=device, dtype=torch.float64)

    print(f"{'Step':>6} {'Samples':>10} {'MSE':>12} {'Mean':>10} {'log_var':>10} {'StdRatio':>10}")
    print("-" * 70)

    t0 = time.perf_counter()
    converged_step = None

    # Warmup: learn mean=x with prior samples first
    print(f"[Warmup: {cfg.num_warmup} steps with prior samples]")
    for step in range(cfg.num_warmup):
        z_batch = sample_prior(cfg.batch_size, device)
        x_batch = simulate(z_batch, cfg.sigma_obs)
        model.update_stats(z_batch, x_batch)  # Update normalization stats
        optimizer.zero_grad()
        loss = model.loss(z_batch, x_batch)
        loss.backward()
        optimizer.step()
        model.update_variance(z_batch, x_batch)
    print(f"[Warmup done, mean(x_obs)={model(x_obs)[0].item():.6f}, log_var={model.log_var.item():.2f}]")

    # Lower learning rate for fine-tuning with proposal
    for pg in optimizer.param_groups:
        pg['lr'] = cfg.lr / 10
    print(f"[LR reduced to {cfg.lr/10} for proposal phase]\n")

    frozen_params = None  # Will hold (mean, std) when proposal is frozen

    for step in range(cfg.num_steps):
        # Check if we should freeze the proposal
        if cfg.prop_freeze > 0 and step == cfg.prop_freeze and frozen_params is None:
            frozen_params = get_proposal_params(model, x_obs, cfg.gamma)
            print(f"[Proposal frozen at step {step}: mean={frozen_params[0]:.6f}, std={frozen_params[1]:.2e}]")

        # Mixed sampling: combine prior and proposal samples
        if cfg.prior_fraction > 0 and step > 0:
            n_prior = int(cfg.batch_size * cfg.prior_fraction)
            n_proposal = cfg.batch_size - n_prior
            z_prior = sample_prior(n_prior, device)
            z_proposal = sample_proposal(model, x_obs, n_proposal, cfg.gamma, device, step, frozen_params)
            z_batch = torch.cat([z_prior, z_proposal], dim=0)
        else:
            z_batch = sample_proposal(model, x_obs, cfg.batch_size, cfg.gamma, device, step, frozen_params)
        x_batch = simulate(z_batch, cfg.sigma_obs)

        # Update normalization stats
        model.update_stats(z_batch, x_batch)

        # Compute importance weights if enabled (only for proposal samples)
        weights = None
        if cfg.iw and step > 0:
            weights = compute_importance_weights(z_batch, model, x_obs, cfg.gamma)

        # Update mean with gradient descent
        optimizer.zero_grad()
        loss = model.loss(z_batch, x_batch, weights=weights)
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
