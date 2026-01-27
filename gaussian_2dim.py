#!/usr/bin/env python3
"""
2-dimensional Gaussian posterior learning with coupled forward model.

Model: f(z) = [z1, z1 * exp(z2)]
       x = f(z) + epsilon, where epsilon ~ N(0, sigma_obs * I)
Prior: z ~ N(0, I)

This creates non-trivial correlations in the posterior.
"""

import argparse
import time
import torch
import torch.nn as nn
import numpy as np


def get_config():
    parser = argparse.ArgumentParser()
    # Problem setup
    parser.add_argument('--x_obs', type=str, default='1,10',
                        help='Observation [x1, x2]')
    parser.add_argument('--sigma_obs', type=float, default=1e-6,
                        help='Observation noise')
    # Training
    parser.add_argument('--num_steps', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_warmup', type=int, default=1000)
    parser.add_argument('--gamma', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--print_every', type=int, default=100)
    # Architecture
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--activation', type=str, default='relu', choices=['relu', 'tanh', 'gelu'])
    parser.add_argument('--momentum', type=float, default=0.01)
    # Learning rates
    parser.add_argument('--lr1', type=float, default=0.01)
    parser.add_argument('--lr2', type=float, default=0.001)
    # Optimizer
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.5)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    # Loss
    parser.add_argument('--rescale_mse', action='store_true', default=True)
    parser.add_argument('--no_rescale_mse', action='store_false', dest='rescale_mse')
    # Covariance
    parser.add_argument('--full_cov', action='store_true', default=False,
                        help='Use full covariance matrix instead of diagonal')
    parser.add_argument('--min_var', type=float, default=1e-20,
                        help='Minimum variance regularization')
    parser.add_argument('--clamp_prior', action='store_true', default=False,
                        help='Clamp eigenvalues of covariance to not exceed prior (1.0)')

    cfg = parser.parse_args()

    # Parse x_obs
    cfg.x_obs_list = [float(x.strip()) for x in cfg.x_obs.split(',')]
    if len(cfg.x_obs_list) != 2:
        raise ValueError(f"x_obs must have exactly 2 values, got {len(cfg.x_obs_list)}")

    return cfg


# =============================================================================
# Forward Model: f(z) = [z1, z1 * exp(z2)]
# =============================================================================

def forward_model(z):
    """f(z) = [z1, z1 * exp(z2)]"""
    z1, z2 = z[:, 0], z[:, 1]
    x1 = z1
    x2 = z1 * torch.exp(z2)
    return torch.stack([x1, x2], dim=1)


def inverse_model(x_obs):
    """z = f^{-1}(x) for scalar observation."""
    x1, x2 = x_obs[0], x_obs[1]
    z1 = x1
    z2 = np.log(x2 / x1)
    return np.array([z1, z2])


def jacobian(z_true):
    """
    Jacobian J = df/dz at z_true.

    J = [[df1/dz1, df1/dz2],   = [[1,           0          ],
         [df2/dz1, df2/dz2]]     [exp(z2), z1*exp(z2)]]
    """
    z1, z2 = z_true[0], z_true[1]
    J = np.array([
        [1.0, 0.0],
        [np.exp(z2), z1 * np.exp(z2)]
    ])
    return J


def expected_covariance(z_true, sigma_obs):
    """
    Expected posterior covariance via linear approximation.

    Cov_z = sigma^2 * (J^T J)^{-1}
    """
    J = jacobian(z_true)
    JTJ = J.T @ J
    JTJ_inv = np.linalg.inv(JTJ)
    return sigma_obs**2 * JTJ_inv


def simulate(z, sigma_obs):
    """Simulate x = f(z) + noise."""
    return forward_model(z) + torch.randn_like(z) * sigma_obs


def sample_prior(n, device):
    """Sample from N(0, I) prior."""
    return torch.randn(n, 2, device=device)


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
    """MLP with online normalization and full covariance tracking."""

    def __init__(self, hidden_dim=128, num_layers=3, activation='relu',
                 momentum=0.01, full_cov=False, min_var=1e-20, clamp_prior=False):
        super().__init__()
        self.ndim = 2
        self.net = build_mlp(2, hidden_dim, 2, num_layers, activation)
        self.momentum = momentum
        self.full_cov = full_cov
        self.min_var = min_var
        self.clamp_prior = clamp_prior

        # Mean vectors
        self.register_buffer('input_mean', torch.zeros(2))
        self.register_buffer('output_mean', torch.zeros(2))

        if full_cov:
            # Full covariance
            self.register_buffer('input_cov_chol', torch.eye(2))
            self.register_buffer('output_cov_chol', torch.eye(2))
            self.register_buffer('residual_cov_chol', torch.eye(2))
            self.register_buffer('input_cov', torch.eye(2))
            self.register_buffer('output_cov', torch.eye(2))
            self.register_buffer('residual_cov', torch.eye(2))
        else:
            # Diagonal
            self.register_buffer('input_std', torch.ones(2))
            self.register_buffer('output_std', torch.ones(2))
            self.register_buffer('log_var', torch.zeros(2))

    def _compute_cov(self, data, mean):
        """Compute covariance matrix from data."""
        centered = data - mean
        n = data.shape[0]
        cov = (centered.T @ centered) / max(n - 1, 1)
        cov = cov + self.min_var * torch.eye(2, device=data.device, dtype=data.dtype)
        return cov

    def _safe_cholesky(self, cov):
        """Compute Cholesky decomposition with fallback."""
        try:
            return torch.linalg.cholesky(cov)
        except RuntimeError:
            cov = cov + 1e-4 * torch.eye(2, device=cov.device, dtype=cov.dtype)
            return torch.linalg.cholesky(cov)

    def _clamp_eigenvalues(self, cov, max_val=1.0):
        """Clamp eigenvalues of covariance matrix to not exceed max_val."""
        eigenvalues, eigenvectors = torch.linalg.eigh(cov)
        eigenvalues = eigenvalues.clamp(min=self.min_var, max=max_val)
        return eigenvectors @ torch.diag(eigenvalues) @ eigenvectors.T

    def update_stats(self, z, x):
        """Update running statistics."""
        with torch.no_grad():
            self.input_mean.lerp_(x.mean(dim=0), self.momentum)
            self.output_mean.lerp_(z.mean(dim=0), self.momentum)

            if self.full_cov:
                batch_input_cov = self._compute_cov(x, self.input_mean)
                batch_output_cov = self._compute_cov(z, self.output_mean)
                self.input_cov.lerp_(batch_input_cov, self.momentum)
                self.output_cov.lerp_(batch_output_cov, self.momentum)
                self.input_cov_chol.copy_(self._safe_cholesky(self.input_cov))
                self.output_cov_chol.copy_(self._safe_cholesky(self.output_cov))
            else:
                self.input_std.lerp_(x.std(dim=0).clamp(min=1e-8), self.momentum)
                self.output_std.lerp_(z.std(dim=0).clamp(min=1e-8), self.momentum)

    def update_variance(self, z, x):
        """Update residual variance/covariance."""
        with torch.no_grad():
            mean = self.forward_mean(x)
            residuals = z - mean

            if self.full_cov:
                batch_cov = self._compute_cov(residuals, torch.zeros(2, device=z.device, dtype=z.dtype))
                self.residual_cov.lerp_(batch_cov, self.momentum)
                # Clamp eigenvalues to prior variance if requested
                if self.clamp_prior:
                    self.residual_cov.copy_(self._clamp_eigenvalues(self.residual_cov, max_val=1.0))
                self.residual_cov_chol.copy_(self._safe_cholesky(self.residual_cov))
            else:
                batch_log_var = torch.log((residuals ** 2).mean(dim=0).clamp(min=1e-20))
                self.log_var.lerp_(batch_log_var, self.momentum)
                # Clamp variance to prior if requested
                if self.clamp_prior:
                    self.log_var.clamp_(max=0.0)  # log(1) = 0

    def forward_mean(self, x):
        """Predict mean."""
        if self.full_cov:
            centered = (x - self.input_mean).T
            x_white = torch.linalg.solve_triangular(
                self.input_cov_chol, centered, upper=False
            ).T
            r = self.net(x_white)
            return self.output_mean + (self.output_cov_chol @ r.T).T
        else:
            x_norm = (x - self.input_mean) / self.input_std.clamp(min=1e-8)
            r = self.net(x_norm)
            return self.output_mean + self.output_std * r

    def get_covariance(self):
        """Get the full 2x2 covariance matrix."""
        if self.full_cov:
            return self.residual_cov.clone()
        else:
            # Diagonal covariance
            var = torch.exp(self.log_var)
            return torch.diag(var)

    def loss(self, z, x, rescale=True):
        """MSE loss, optionally rescaled."""
        mean = self.forward_mean(x)
        residuals = z - mean

        if rescale:
            if self.full_cov:
                L = self.residual_cov_chol.detach()
                r_white = torch.linalg.solve_triangular(L, residuals.T, upper=False).T
                return (r_white ** 2).mean()
            else:
                var = torch.exp(self.log_var).detach().clamp(min=1e-20)
                return ((residuals ** 2) / var).mean()
        else:
            return (residuals ** 2).mean()

    def log_prob(self, z, x):
        """Gaussian log probability."""
        mean = self.forward_mean(x)
        residuals = z - mean

        if self.full_cov:
            L = self.residual_cov_chol
            log_det = 2 * torch.log(torch.diag(L)).sum()
            r_white = torch.linalg.solve_triangular(L, residuals.T, upper=False).T
            mahal = (r_white ** 2).sum(dim=-1)
            return -0.5 * (2 * np.log(2 * np.pi) + log_det + mahal)
        else:
            var = torch.exp(self.log_var)
            return -0.5 * (np.log(2 * np.pi) + self.log_var + (residuals ** 2) / var).sum(dim=-1)

    def sample(self, x, temperature=1.0):
        """Sample from predicted Gaussian."""
        mean = self.forward_mean(x)
        eps = torch.randn_like(mean)

        if self.full_cov:
            L = self.residual_cov_chol * np.sqrt(temperature)
            return mean + (L @ eps.T).T
        else:
            std = torch.exp(0.5 * self.log_var) * np.sqrt(temperature)
            return mean + std * eps


def sample_proposal(model, x_obs, n, gamma, device, step):
    """Sample from proposal."""
    with torch.no_grad():
        if step == 0:
            return sample_prior(n, device)
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

    # Compute ground truth
    x_obs_np = np.array(cfg.x_obs_list)
    z_true = inverse_model(x_obs_np)
    cov_expected = expected_covariance(z_true, cfg.sigma_obs)

    print(f"Device: {device}")
    print(f"Forward model: f(z) = [z1, z1 * exp(z2)]")
    print(f"x_obs: {cfg.x_obs_list}")
    print(f"z_true: [{z_true[0]:.4f}, {z_true[1]:.4f}]")
    print(f"sigma_obs: {cfg.sigma_obs:.0e}")
    print(f"Expected covariance:")
    print(f"  var(z1)={cov_expected[0,0]:.2e}, var(z2)={cov_expected[1,1]:.2e}, cov(z1,z2)={cov_expected[0,1]:.2e}")
    print(f"Training: {cfg.num_steps} steps x {cfg.batch_size} = {cfg.num_steps * cfg.batch_size} samples")
    print()

    # Create model
    model = NormalizedMLP(
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        activation=cfg.activation,
        momentum=cfg.momentum,
        full_cov=cfg.full_cov,
        min_var=cfg.min_var,
        clamp_prior=cfg.clamp_prior,
    ).double().to(device)

    cov_mode = "full" if cfg.full_cov else "diagonal"
    clamp_str = " (eigenvalues clamped to prior)" if cfg.clamp_prior else ""
    print(f"Using {cov_mode} covariance{clamp_str}")
    print()

    optimizer = torch.optim.AdamW(
        model.net.parameters(),
        lr=cfg.lr1,
        betas=(cfg.beta1, cfg.beta2),
        weight_decay=cfg.weight_decay,
    )

    x_obs = torch.tensor([cfg.x_obs_list], device=device, dtype=torch.float64)

    # Print header
    print(f"{'Step':>6} {'Samples':>10} {'Loss':>12} {'var1_r':>8} {'var2_r':>8} {'cov12_r':>8}")
    print("-" * 62)

    t0 = time.perf_counter()

    # Warmup
    print(f"[Warmup: {cfg.num_warmup} steps with prior samples]")
    for step in range(cfg.num_warmup):
        z_batch = sample_prior(cfg.batch_size, device)
        x_batch = simulate(z_batch, cfg.sigma_obs)
        model.update_stats(z_batch, x_batch)
        optimizer.zero_grad()
        loss = model.loss(z_batch, x_batch, rescale=cfg.rescale_mse)
        loss.backward()
        optimizer.step()
        model.update_variance(z_batch, x_batch)

    mean_pred = model.forward_mean(x_obs)
    print(f"[Warmup done, mean(x_obs)={mean_pred.squeeze().tolist()}]")

    for pg in optimizer.param_groups:
        pg['lr'] = cfg.lr2
    print(f"[LR changed to {cfg.lr2} for sequential phase]\n")

    # Sequential phase
    for step in range(cfg.num_steps):
        z_batch = sample_proposal(model, x_obs, cfg.batch_size, cfg.gamma, device, step)
        x_batch = simulate(z_batch, cfg.sigma_obs)

        model.update_stats(z_batch, x_batch)
        optimizer.zero_grad()
        loss = model.loss(z_batch, x_batch, rescale=cfg.rescale_mse)
        loss.backward()
        optimizer.step()
        model.update_variance(z_batch, x_batch)

        if step % cfg.print_every == 0 or step == cfg.num_steps - 1:
            cov_pred = model.get_covariance().detach().cpu().numpy()
            var1_r = cov_pred[0, 0] / cov_expected[0, 0]
            var2_r = cov_pred[1, 1] / cov_expected[1, 1]
            cov12_r = cov_pred[0, 1] / cov_expected[0, 1] if abs(cov_expected[0, 1]) > 1e-30 else np.nan
            samples = (step + 1) * cfg.batch_size

            print(f"{step:>6} {samples:>10} {loss.item():>12.2e} {var1_r:>8.3f} {var2_r:>8.3f} {cov12_r:>8.3f}")

    elapsed = time.perf_counter() - t0

    # Final results
    mean_pred = model.forward_mean(x_obs).squeeze().detach().cpu().numpy()
    cov_pred = model.get_covariance().detach().cpu().numpy()

    print()
    print("Final results:")
    print(f"  Mean: [{mean_pred[0]:.6f}, {mean_pred[1]:.6f}]")
    print(f"  True: [{z_true[0]:.6f}, {z_true[1]:.6f}]")
    print(f"  Error: [{abs(mean_pred[0]-z_true[0]):.2e}, {abs(mean_pred[1]-z_true[1]):.2e}]")
    print()
    print(f"  Predicted covariance:")
    print(f"    [[{cov_pred[0,0]:.2e}, {cov_pred[0,1]:.2e}],")
    print(f"     [{cov_pred[1,0]:.2e}, {cov_pred[1,1]:.2e}]]")
    print(f"  Expected covariance:")
    print(f"    [[{cov_expected[0,0]:.2e}, {cov_expected[0,1]:.2e}],")
    print(f"     [{cov_expected[1,0]:.2e}, {cov_expected[1,1]:.2e}]]")
    print()
    print(f"  Ratios: var1={cov_pred[0,0]/cov_expected[0,0]:.3f}, "
          f"var2={cov_pred[1,1]/cov_expected[1,1]:.3f}, "
          f"cov12={cov_pred[0,1]/cov_expected[0,1]:.3f}")
    print(f"\nTime: {elapsed:.2f}s")


if __name__ == "__main__":
    main()
