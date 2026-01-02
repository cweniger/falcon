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
from adam import TrackingAdam

def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sigma_obs', type=float, default=1e-6)
    parser.add_argument('--x_obs', type=float, default=0.3)
    parser.add_argument('--num_steps', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_warmup', type=int, default=1000)
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
    parser.add_argument('--model', type=str, default='identity', choices=['identity', 'exp', 'cubic'],
                        help='Forward model: identity (x=z), exp (x=exp(z)), cubic (x=z^3)')
    # Architecture options
    parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden layer dimension')
    parser.add_argument('--num_layers', type=int, default=3, help='Number of hidden layers')
    parser.add_argument('--activation', type=str, default='relu', choices=['relu', 'tanh', 'gelu'],
                        help='Activation function')
    parser.add_argument('--momentum', type=float, default=0.01, help='EMA momentum for normalization stats')
    # Learning rate options
    parser.add_argument('--lr1', type=float, default=0.01, help='Learning rate for warmup phase')
    parser.add_argument('--lr2', type=float, default=0.001, help='Learning rate for sequential phase')
    # Plotting options
    parser.add_argument('--residual_plot', type=str, default=None,
                        help='Save residual plot to this file (e.g., residuals.png)')
    # Loss options
    parser.add_argument('--rescale_mse', action='store_true',
                        help='Divide MSE by variance (detached) for stable gradients with narrow posteriors')
    # Optimizer options
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'tracking_adam'],
                        help='Optimizer: adam or tracking_adam (adds diffusion for plasticity)')
    parser.add_argument('--diffusion_scale', type=float, default=0.0001,
                        help='TrackingAdam: diffusion noise scale (typical 0.0001-0.001)')
    parser.add_argument('--momentum_gating', action='store_true', default=True,
                        help='TrackingAdam: suppress noise when momentum is large')
    parser.add_argument('--no_momentum_gating', action='store_false', dest='momentum_gating',
                        help='TrackingAdam: disable momentum gating')
    return parser.parse_args()


def get_activation(name):
    """Get activation function by name."""
    if name == 'relu':
        return nn.ReLU()
    elif name == 'tanh':
        return nn.Tanh()
    elif name == 'gelu':
        return nn.GELU()
    else:
        raise ValueError(f"Unknown activation: {name}")


def build_mlp(input_dim, hidden_dim, output_dim, num_layers, activation):
    """Build MLP with configurable depth and activation."""
    layers = []
    layers.append(nn.Linear(input_dim, hidden_dim))
    layers.append(get_activation(activation))
    for _ in range(num_layers - 1):
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        layers.append(get_activation(activation))
    layers.append(nn.Linear(hidden_dim, output_dim))
    return nn.Sequential(*layers)


class GaussianMLP(nn.Module):
    """MLP learns mean, variance is computed from residuals (closed-form optimal)."""

    def __init__(self, hidden_dim=64, num_layers=3, activation='relu', momentum=0.01):
        super().__init__()
        self.net = build_mlp(1, hidden_dim, 1, num_layers, activation)
        # Running estimate of variance (EMA)
        self.register_buffer('log_var', torch.tensor(0.0))
        self.var_momentum = momentum

    def forward(self, x):
        mean = self.net(x)
        return mean, self.log_var.expand(x.shape[0], 1)

    def update_stats(self, z, x, weights=None):
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

    def loss(self, z, x, weights=None, rescale=False):
        """MSE loss for mean (variance is updated separately)."""
        mean = self.net(x)
        squared_errors = (z - mean) ** 2
        if rescale:
            var = torch.exp(self.log_var).detach() + 1e-20
            squared_errors = squared_errors / var
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

    def __init__(self, hidden_dim=64, num_layers=3, activation='relu', momentum=0.01):
        super().__init__()
        self.net = build_mlp(1, hidden_dim, 1, num_layers, activation)
        # Input normalization (running stats)
        self.register_buffer('input_mean', torch.tensor(0.0))
        self.register_buffer('input_std', torch.tensor(1.0))
        # Output denormalization (running stats)
        self.register_buffer('output_mean', torch.tensor(0.0))
        self.register_buffer('output_std', torch.tensor(1.0))
        # Variance estimate
        self.register_buffer('log_var', torch.tensor(0.0))
        self.var_momentum = momentum

    def update_stats(self, z, x, weights=None):
        """Update running input/output statistics (optionally weighted for IW)."""
        with torch.no_grad():
            if weights is not None:
                # Weighted statistics for importance weighting
                w = weights.view(-1, 1)
                batch_mean_x = (w * x).sum()
                batch_mean_z = (w * z).sum()
                # Weighted std: sqrt(sum(w * (x - mean)^2))
                batch_std_x = torch.sqrt((w * (x - batch_mean_x)**2).sum()) + 1e-8
                batch_std_z = torch.sqrt((w * (z - batch_mean_z)**2).sum()) + 1e-8
            else:
                batch_mean_x = x.mean()
                batch_std_x = x.std() + 1e-8
                batch_mean_z = z.mean()
                batch_std_z = z.std() + 1e-8

            self.input_mean.copy_((1 - self.var_momentum) * self.input_mean + self.var_momentum * batch_mean_x)
            self.input_std.copy_((1 - self.var_momentum) * self.input_std + self.var_momentum * batch_std_x)
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

    def loss(self, z, x, weights=None, rescale=False):
        """MSE loss for mean."""
        mean, log_var = self(x)
        squared_errors = (z - mean) ** 2
        if rescale:
            var = torch.exp(log_var).detach() + 1e-8
            squared_errors = squared_errors / var
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


def forward_model(z, model_type):
    """Apply forward model f(z)."""
    if model_type == 'identity':
        return z
    elif model_type == 'exp':
        return torch.exp(z)
    elif model_type == 'cubic':
        return z ** 3
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def inverse_model(x, model_type):
    """Apply inverse model f^{-1}(x) to get true z from observed x."""
    if model_type == 'identity':
        return x
    elif model_type == 'exp':
        return np.log(x)
    elif model_type == 'cubic':
        return np.sign(x) * np.abs(x) ** (1/3)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def forward_derivative(z, model_type):
    """Compute f'(z) for the forward model."""
    if model_type == 'identity':
        return 1.0
    elif model_type == 'exp':
        return np.exp(z)
    elif model_type == 'cubic':
        return 3 * z ** 2
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def simulate(z, sigma_obs, model_type='identity'):
    """Simulate x = f(z) + noise."""
    return forward_model(z, model_type) + torch.randn_like(z) * sigma_obs

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
    np.random.seed(cfg.seed)
    torch.set_default_dtype(torch.float64)
    device = torch.device(cfg.device if torch.cuda.is_available() else 'cpu')

    print(f"Device: {device}")
    print(f"Forward model: x = {cfg.model}(z) + noise")
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
        model = NormalizedMLP(
            hidden_dim=cfg.hidden_dim, num_layers=cfg.num_layers,
            activation=cfg.activation, momentum=cfg.momentum
        ).double().to(device)
    else:
        model = GaussianMLP(
            hidden_dim=cfg.hidden_dim, num_layers=cfg.num_layers,
            activation=cfg.activation, momentum=cfg.momentum
        ).double().to(device)

    # Create optimizer
    if cfg.optimizer == 'tracking_adam':
        optimizer = TrackingAdam(
            model.net.parameters(), lr=cfg.lr1,
            diffusion_scale=cfg.diffusion_scale,
            momentum_gating=cfg.momentum_gating
        )
        print(f"Using TrackingAdam (diffusion={cfg.diffusion_scale}, gating={cfg.momentum_gating})")
    else:
        optimizer = torch.optim.Adam(model.net.parameters(), lr=cfg.lr1)
    x_obs = torch.tensor([[cfg.x_obs]], device=device, dtype=torch.float64)
    z_true = inverse_model(cfg.x_obs, cfg.model)  # True z that produces x_obs
    f_prime = forward_derivative(z_true, cfg.model)  # f'(z_true)
    expected_std = cfg.sigma_obs / abs(f_prime)  # Expected posterior std

    print(f"x_obs = {cfg.x_obs}, z_true = {z_true:.6f}, f'(z_true) = {f_prime:.4f}")
    print(f"Expected posterior std = sigma_obs / |f'| = {expected_std:.2e}")
    print(f"{'Step':>6} {'Samples':>10} {'MSE':>12} {'Mean':>10} {'log_var':>10} {'StdRatio':>10}")
    print("-" * 70)

    t0 = time.perf_counter()
    converged_step = None

    # Warmup: learn mean=x with prior samples first
    print(f"[Warmup: {cfg.num_warmup} steps with prior samples]")
    for step in range(cfg.num_warmup):
        z_batch = sample_prior(cfg.batch_size, device)
        x_batch = simulate(z_batch, cfg.sigma_obs, cfg.model)
        model.update_stats(z_batch, x_batch)  # Update normalization stats
        optimizer.zero_grad()
        loss = model.loss(z_batch, x_batch, rescale=cfg.rescale_mse)
        loss.backward()
        optimizer.step()
        model.update_variance(z_batch, x_batch)
    print(f"[Warmup done, mean(x_obs)={model(x_obs)[0].item():.6f}, log_var={model.log_var.item():.2f}]")

    # Lower learning rate for fine-tuning with proposal
    for pg in optimizer.param_groups:
        pg['lr'] = cfg.lr2
    print(f"[LR changed to {cfg.lr2} for sequential phase]\n")

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
        x_batch = simulate(z_batch, cfg.sigma_obs, cfg.model)

        # Compute importance weights if enabled (must be before update_stats)
        weights = None
        if cfg.iw and step > 0:
            weights = compute_importance_weights(z_batch, model, x_obs, cfg.gamma)

        # Update normalization stats (with weights if IW enabled)
        model.update_stats(z_batch, x_batch, weights)

        # Update mean with gradient descent
        optimizer.zero_grad()
        loss = model.loss(z_batch, x_batch, weights=weights, rescale=cfg.rescale_mse)
        loss.backward()
        optimizer.step()

        # Update variance with closed-form optimal
        model.update_variance(z_batch, x_batch)

        if step % cfg.print_every == 0 or step == cfg.num_steps - 1:
            mean_pred = model(x_obs)[0].item()
            log_var = model.log_var.item()
            std_pred = np.exp(0.5 * log_var)
            std_ratio = std_pred / expected_std
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

    print(f"\nFinal: mean={mean_pred:.6f} (true={z_true:.6f}, error={abs(mean_pred-z_true):.2e})")
    print(f"       std={std_pred:.2e} (expected={expected_std:.2e})")
    print(f"       std_ratio={std_pred/expected_std:.3f}")
    print(f"\nTime: {elapsed:.2f}s")

    # Save residual plot if requested
    if cfg.residual_plot:
        save_residual_plot(model, cfg, x_obs, device, cfg.residual_plot)


def save_residual_plot(model, cfg, x_obs, device, filename):
    """Generate scatter plot of true vs predicted posterior modes."""
    import matplotlib.pyplot as plt

    with torch.no_grad():
        # Sample 128 z from proposal (using current model)
        z_samples = model.sample(x_obs.expand(128, -1), temperature=1.0/cfg.gamma)
        z_samples = z_samples.unsqueeze(-1)

        # Simulate x for each z
        x_samples = simulate(z_samples, cfg.sigma_obs, cfg.model)

        # True posterior mode: z_true = f^{-1}(x) for each x
        z_true = torch.tensor([inverse_model(x.item(), cfg.model) for x in x_samples],
                              device=device, dtype=torch.float64)

        # Predicted posterior mode: model(x)[0]
        z_pred = model(x_samples)[0].squeeze()

    # Convert to numpy for plotting
    z_true_np = z_true.cpu().numpy()
    z_pred_np = z_pred.cpu().numpy()

    # Create scatter plot
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(z_true_np, z_pred_np, alpha=0.5, s=20)

    # Add diagonal line
    lims = [min(z_true_np.min(), z_pred_np.min()),
            max(z_true_np.max(), z_pred_np.max())]
    ax.plot(lims, lims, 'k--', alpha=0.5, label='y=x')

    ax.set_xlabel('True posterior mode (inverse model)')
    ax.set_ylabel('Predicted posterior mode (MLP)')
    ax.set_title(f'Residual plot: {cfg.model} model, σ={cfg.sigma_obs:.0e}')
    ax.legend()
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"[Residual plot saved to {filename}]")


if __name__ == "__main__":
    main()
