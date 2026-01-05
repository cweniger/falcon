#!/usr/bin/env python3
"""
N-dimensional Gaussian posterior learning with component-wise forward models.

Model: x_i = f_i(z_i) + epsilon_i, where epsilon_i ~ N(0, sigma_obs)
Prior: z ~ N(0, I)

The code supports component-wise functions f_i(z_i), but is structured to
trivially extend to non-component-wise functions f(z)_i.
"""

import argparse
import time
import torch
import torch.nn as nn
import numpy as np


# =============================================================================
# Forward Model Registry
# =============================================================================
# Each model: (forward, inverse, derivative)
# All functions work element-wise on tensors

MODELS = {
    'identity': (
        lambda z: z,
        lambda x: x,
        lambda z: torch.ones_like(z),
    ),
    'exp': (
        lambda z: torch.exp(z),
        lambda x: torch.log(x),
        lambda z: torch.exp(z),
    ),
    'cubic': (
        lambda z: z ** 3,
        lambda x: torch.sign(x) * torch.abs(x) ** (1/3),
        lambda z: 3 * z ** 2,
    ),
    'quintic': (
        lambda z: z ** 5,
        lambda x: torch.sign(x) * torch.abs(x) ** (1/5),
        lambda z: 5 * z ** 4,
    ),
}


def get_config():
    parser = argparse.ArgumentParser()
    # Problem setup
    parser.add_argument('--models', type=str, default='identity,exp,cubic',
                        help='Comma-separated list of forward models per dimension')
    parser.add_argument('--x_obs', type=str, default=None,
                        help='Comma-separated observation values (default: 1.0 each)')
    parser.add_argument('--sigma_obs', type=float, default=1e-3,
                        help='Observation noise (same for all components)')
    # Training
    parser.add_argument('--num_steps', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_warmup', type=int, default=1000)
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--print_every', type=int, default=100)
    # Architecture
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden layer dimension')
    parser.add_argument('--num_layers', type=int, default=3, help='Number of hidden layers')
    parser.add_argument('--activation', type=str, default='relu', choices=['relu', 'tanh', 'gelu'])
    parser.add_argument('--momentum', type=float, default=0.01, help='EMA momentum for normalization')
    # Learning rates
    parser.add_argument('--lr1', type=float, default=0.01, help='Learning rate for warmup')
    parser.add_argument('--lr2', type=float, default=0.001, help='Learning rate for sequential phase')
    # Optimizer (good defaults from gaussian_test.py)
    parser.add_argument('--beta1', type=float, default=0.5, help='AdamW beta1')
    parser.add_argument('--beta2', type=float, default=0.5, help='AdamW beta2')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='AdamW weight decay')
    # Loss
    parser.add_argument('--rescale_mse', action='store_true', default=True,
                        help='Divide MSE by variance (default: on)')
    parser.add_argument('--no_rescale_mse', action='store_false', dest='rescale_mse')

    cfg = parser.parse_args()

    # Parse models list
    cfg.model_names = [m.strip() for m in cfg.models.split(',')]
    cfg.ndim = len(cfg.model_names)

    # Parse x_obs (default: 1.0 for each dimension)
    if cfg.x_obs is None:
        cfg.x_obs_list = [1.0] * cfg.ndim
    else:
        cfg.x_obs_list = [float(x.strip()) for x in cfg.x_obs.split(',')]
        if len(cfg.x_obs_list) != cfg.ndim:
            raise ValueError(f"x_obs has {len(cfg.x_obs_list)} values but {cfg.ndim} models specified")

    return cfg


# =============================================================================
# Network
# =============================================================================

def get_activation(name):
    return {'relu': nn.ReLU, 'tanh': nn.Tanh, 'gelu': nn.GELU}[name]()


def build_mlp(input_dim, hidden_dim, output_dim, num_layers, activation):
    layers = [nn.Linear(input_dim, hidden_dim), get_activation(activation)]
    for _ in range(num_layers - 1):
        layers.extend([nn.Linear(hidden_dim, hidden_dim), get_activation(activation)])
    layers.append(nn.Linear(hidden_dim, output_dim))
    return nn.Sequential(*layers)


class NormalizedMLP(nn.Module):
    """MLP with online input/output normalization for N-dimensional problems.

    Prediction: mu(x) = output_mean + output_std * mlp((x - input_mean) / input_std)
    """

    def __init__(self, ndim, hidden_dim=128, num_layers=3, activation='relu', momentum=0.01):
        super().__init__()
        self.ndim = ndim
        self.net = build_mlp(ndim, hidden_dim, ndim, num_layers, activation)
        self.momentum = momentum

        # Per-dimension normalization stats
        self.register_buffer('input_mean', torch.zeros(ndim))
        self.register_buffer('input_std', torch.ones(ndim))
        self.register_buffer('output_mean', torch.zeros(ndim))
        self.register_buffer('output_std', torch.ones(ndim))
        # Per-dimension log variance
        self.register_buffer('log_var', torch.zeros(ndim))

    def update_stats(self, z, x):
        """Update running input/output statistics."""
        with torch.no_grad():
            # Input stats (from x)
            self.input_mean.lerp_(x.mean(dim=0), self.momentum)
            self.input_std.lerp_(x.std(dim=0).clamp(min=1e-8), self.momentum)
            # Output stats (from z)
            self.output_mean.lerp_(z.mean(dim=0), self.momentum)
            self.output_std.lerp_(z.std(dim=0).clamp(min=1e-8), self.momentum)

    def update_variance(self, z, x):
        """Update per-dimension variance estimate from residuals."""
        with torch.no_grad():
            mean = self.forward_mean(x)
            residuals_sq = (z - mean) ** 2
            batch_log_var = torch.log(residuals_sq.mean(dim=0).clamp(min=1e-20))
            self.log_var.lerp_(batch_log_var, self.momentum)

    def forward_mean(self, x):
        """Predict mean."""
        x_norm = (x - self.input_mean) / self.input_std.clamp(min=1e-8)
        r = self.net(x_norm)
        return self.output_mean + self.output_std * r

    def forward(self, x):
        """Return (mean, log_var) for batch."""
        mean = self.forward_mean(x)
        log_var = self.log_var.expand(x.shape[0], -1)
        return mean, log_var

    def loss(self, z, x, rescale=True):
        """MSE loss, optionally rescaled by per-dim variance."""
        mean = self.forward_mean(x)
        sq_errors = (z - mean) ** 2
        if rescale:
            var = torch.exp(self.log_var).detach().clamp(min=1e-20)
            sq_errors = sq_errors / var
        return sq_errors.mean()

    def log_prob(self, z, x):
        """Gaussian log probability (factorized over dimensions)."""
        mean, log_var = self(x)
        var = torch.exp(log_var)
        return -0.5 * (np.log(2 * np.pi) + log_var + (z - mean) ** 2 / var).sum(dim=-1)

    def sample(self, x, temperature=1.0):
        """Sample from predicted Gaussian."""
        mean, log_var = self(x)
        std = torch.exp(0.5 * log_var) * np.sqrt(temperature)
        return mean + std * torch.randn_like(mean)


# =============================================================================
# Forward Model Functions
# =============================================================================

class ForwardModel:
    """Wrapper for component-wise forward models.

    Structured to support non-component-wise f(z) in the future.
    """

    def __init__(self, model_names):
        self.model_names = model_names
        self.ndim = len(model_names)
        self.models = [MODELS[name] for name in model_names]

    def forward(self, z):
        """Apply forward model: x = f(z). Shape: (batch, ndim) -> (batch, ndim)"""
        x = torch.zeros_like(z)
        for i, (fwd, _, _) in enumerate(self.models):
            x[:, i] = fwd(z[:, i])
        return x

    def inverse(self, x):
        """Apply inverse model: z = f^{-1}(x). Shape: (batch, ndim) -> (batch, ndim)"""
        z = torch.zeros_like(x)
        for i, (_, inv, _) in enumerate(self.models):
            z[:, i] = inv(x[:, i])
        return z

    def derivative(self, z):
        """Compute f'(z) per dimension. Shape: (batch, ndim) -> (batch, ndim)"""
        d = torch.zeros_like(z)
        for i, (_, _, deriv) in enumerate(self.models):
            d[:, i] = deriv(z[:, i])
        return d

    def inverse_scalar(self, x_vals):
        """Inverse for scalar values (for computing z_true from x_obs)."""
        z_vals = []
        for i, (_, inv, _) in enumerate(self.models):
            x_t = torch.tensor([x_vals[i]], dtype=torch.float64)
            z_vals.append(inv(x_t).item())
        return z_vals

    def derivative_scalar(self, z_vals):
        """Derivative at scalar values."""
        d_vals = []
        for i, (_, _, deriv) in enumerate(self.models):
            z_t = torch.tensor([z_vals[i]], dtype=torch.float64)
            d_vals.append(deriv(z_t).item())
        return d_vals


def simulate(z, sigma_obs, fwd_model):
    """Simulate x = f(z) + noise."""
    return fwd_model.forward(z) + torch.randn_like(z) * sigma_obs


def sample_prior(n, ndim, device):
    """Sample from N(0, I) prior."""
    return torch.randn(n, ndim, device=device)


def sample_proposal(model, x_obs, n, gamma, device, step):
    """Sample from proposal (prior at step 0, else from model)."""
    with torch.no_grad():
        if step == 0:
            return sample_prior(n, x_obs.shape[-1], device)
        temperature = 1.0 / gamma
        x_batch = x_obs.expand(n, -1)
        return model.sample(x_batch, temperature=temperature)


# =============================================================================
# Main
# =============================================================================

def main():
    cfg = get_config()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.set_default_dtype(torch.float64)
    device = torch.device(cfg.device if torch.cuda.is_available() else 'cpu')

    # Setup forward model
    fwd_model = ForwardModel(cfg.model_names)

    # Compute ground truth
    z_true = fwd_model.inverse_scalar(cfg.x_obs_list)
    f_prime = fwd_model.derivative_scalar(z_true)
    expected_std = [cfg.sigma_obs / abs(fp) for fp in f_prime]

    print(f"Device: {device}")
    print(f"Dimensions: {cfg.ndim}")
    print(f"Models: {cfg.model_names}")
    print(f"x_obs: {cfg.x_obs_list}")
    print(f"z_true: {[f'{z:.4f}' for z in z_true]}")
    print(f"f'(z_true): {[f'{fp:.4f}' for fp in f_prime]}")
    print(f"Expected std: {[f'{s:.2e}' for s in expected_std]}")
    print(f"sigma_obs: {cfg.sigma_obs:.0e}")
    print(f"Training: {cfg.num_steps} steps x {cfg.batch_size} = {cfg.num_steps * cfg.batch_size} samples")
    print()

    # Create model
    model = NormalizedMLP(
        ndim=cfg.ndim,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        activation=cfg.activation,
        momentum=cfg.momentum,
    ).double().to(device)

    # Create optimizer (good defaults: AdamW with betas=(0.5, 0.5))
    optimizer = torch.optim.AdamW(
        model.net.parameters(),
        lr=cfg.lr1,
        betas=(cfg.beta1, cfg.beta2),
        weight_decay=cfg.weight_decay,
    )

    x_obs = torch.tensor([cfg.x_obs_list], device=device, dtype=torch.float64)

    # Print header
    print(f"{'Step':>6} {'Samples':>10} {'Loss':>12} " +
          " ".join([f"std_r[{i}]" for i in range(cfg.ndim)]))
    print("-" * (30 + 10 * cfg.ndim))

    t0 = time.perf_counter()

    # Warmup phase
    print(f"[Warmup: {cfg.num_warmup} steps with prior samples]")
    for step in range(cfg.num_warmup):
        z_batch = sample_prior(cfg.batch_size, cfg.ndim, device)
        x_batch = simulate(z_batch, cfg.sigma_obs, fwd_model)
        model.update_stats(z_batch, x_batch)
        optimizer.zero_grad()
        loss = model.loss(z_batch, x_batch, rescale=cfg.rescale_mse)
        loss.backward()
        optimizer.step()
        model.update_variance(z_batch, x_batch)

    mean_pred, _ = model(x_obs)
    print(f"[Warmup done, mean(x_obs)={mean_pred.squeeze().tolist()}]")

    # Lower LR for sequential phase
    for pg in optimizer.param_groups:
        pg['lr'] = cfg.lr2
    print(f"[LR changed to {cfg.lr2} for sequential phase]\n")

    # Sequential phase
    for step in range(cfg.num_steps):
        z_batch = sample_proposal(model, x_obs, cfg.batch_size, cfg.gamma, device, step)
        x_batch = simulate(z_batch, cfg.sigma_obs, fwd_model)

        model.update_stats(z_batch, x_batch)
        optimizer.zero_grad()
        loss = model.loss(z_batch, x_batch, rescale=cfg.rescale_mse)
        loss.backward()
        optimizer.step()
        model.update_variance(z_batch, x_batch)

        if step % cfg.print_every == 0 or step == cfg.num_steps - 1:
            mean_pred, log_var = model(x_obs)
            std_pred = torch.exp(0.5 * log_var).squeeze()
            std_ratios = [std_pred[i].item() / expected_std[i] for i in range(cfg.ndim)]
            samples = (step + 1) * cfg.batch_size

            ratio_str = " ".join([f"{r:>8.2f}" for r in std_ratios])
            print(f"{step:>6} {samples:>10} {loss.item():>12.2e} {ratio_str}")

    elapsed = time.perf_counter() - t0

    # Final results
    mean_pred, log_var = model(x_obs)
    mean_pred = mean_pred.squeeze()
    std_pred = torch.exp(0.5 * log_var).squeeze()

    print()
    print("Final results:")
    for i in range(cfg.ndim):
        z_err = abs(mean_pred[i].item() - z_true[i])
        std_ratio = std_pred[i].item() / expected_std[i]
        print(f"  dim {i} ({cfg.model_names[i]:>8}): "
              f"mean={mean_pred[i].item():.6f} (true={z_true[i]:.6f}, err={z_err:.2e}), "
              f"std={std_pred[i].item():.2e} (exp={expected_std[i]:.2e}), "
              f"ratio={std_ratio:.3f}")

    print(f"\nTime: {elapsed:.2f}s")


if __name__ == "__main__":
    main()
