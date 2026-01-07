#!/usr/bin/env python3
"""
2-dimensional Gaussian posterior learning with coupled forward model.
Full covariance version with eigenvalue-based tempered likelihood proposal.

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
    # Covariance
    parser.add_argument('--min_var', type=float, default=1e-20,
                        help='Minimum variance regularization')
    parser.add_argument('--eig_update_freq', type=int, default=1,
                        help='Frequency of eigendecomposition updates')

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
    """MLP with online normalization and eigenvalue-based covariance tracking."""

    def __init__(self, hidden_dim=128, num_layers=3, activation='relu',
                 momentum=0.01, min_var=1e-20, eig_update_freq=1):
        super().__init__()
        self.ndim = 2
        self.net = build_mlp(2, hidden_dim, 2, num_layers, activation)
        self.momentum = momentum
        self.min_var = min_var
        self.eig_update_freq = eig_update_freq
        self.step_counter = 0

        # Mean vectors
        self.register_buffer('input_mean', torch.zeros(2))
        self.register_buffer('output_mean', torch.zeros(2))

        # Covariance matrices
        self.register_buffer('input_cov', torch.eye(2))
        self.register_buffer('output_cov', torch.eye(2))
        self.register_buffer('residual_cov', torch.eye(2))

        # Cholesky for input/output normalization (kept for efficiency)
        self.register_buffer('input_cov_chol', torch.eye(2))
        self.register_buffer('output_cov_chol', torch.eye(2))

        # Eigendecomposition of residual covariance
        self.register_buffer('residual_eigvals', torch.ones(2))
        self.register_buffer('residual_eigvecs', torch.eye(2))

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

    def _update_eigendecomp(self):
        """Update eigendecomposition of residual covariance."""
        eigvals, eigvecs = torch.linalg.eigh(self.residual_cov)
        # Clamp eigenvalues for numerical stability
        eigvals = eigvals.clamp(min=self.min_var)
        self.residual_eigvals.copy_(eigvals)
        self.residual_eigvecs.copy_(eigvecs)

    def update_stats(self, z, x):
        """Update running statistics."""
        with torch.no_grad():
            self.input_mean.lerp_(x.mean(dim=0), self.momentum)
            self.output_mean.lerp_(z.mean(dim=0), self.momentum)

            batch_input_cov = self._compute_cov(x, self.input_mean)
            batch_output_cov = self._compute_cov(z, self.output_mean)
            self.input_cov.lerp_(batch_input_cov, self.momentum)
            self.output_cov.lerp_(batch_output_cov, self.momentum)
            self.input_cov_chol.copy_(self._safe_cholesky(self.input_cov))
            self.output_cov_chol.copy_(self._safe_cholesky(self.output_cov))

    def update_covariance(self, z, x):
        """Update residual covariance and eigendecomposition."""
        with torch.no_grad():
            mean = self.forward_mean(x)
            residuals = z - mean

            batch_cov = self._compute_cov(residuals, torch.zeros(2, device=z.device, dtype=z.dtype))
            self.residual_cov.lerp_(batch_cov, self.momentum)

            # Update eigendecomposition at specified frequency
            self.step_counter += 1
            if self.step_counter % self.eig_update_freq == 0:
                self._update_eigendecomp()

    def forward_mean(self, x):
        """Predict mean using Cholesky-based whitening."""
        centered = (x - self.input_mean).T
        x_white = torch.linalg.solve_triangular(
            self.input_cov_chol, centered, upper=False
        ).T
        r = self.net(x_white)
        return self.output_mean + (self.output_cov_chol @ r.T).T

    def get_covariance(self):
        """Get the full 2x2 covariance matrix."""
        return self.residual_cov.clone()

    def log_prob(self, z, x):
        """Gaussian log probability using eigendecomposition.

        Returns per-sample log probabilities.
        log_det is detached since covariance is updated separately via EMA.
        """
        mean = self.forward_mean(x)
        residuals = z - mean

        # log|Σ| = sum(log(d_i))
        log_det = torch.log(self.residual_eigvals).sum()

        # Mahalanobis via eigenbasis: sum_i (r_i^2 / d_i) where r_i = (V^T @ residuals)_i
        V = self.residual_eigvecs
        d = self.residual_eigvals
        r_proj = V.T @ residuals.T  # (ndim, batch)
        mahal = (r_proj ** 2 / d.unsqueeze(1)).sum(dim=0)  # (batch,)

        return -0.5 * (self.ndim * np.log(2 * np.pi) + log_det.detach() + mahal)

    def sample(self, x, gamma=1.0):
        """Sample from proposal = tempered_likelihood × prior.

        The proposal precision is: gamma * lambda_like + 1
        where lambda_like = max(1/d - 1, 0) and d are posterior covariance eigenvalues.
        """
        mean = self.forward_mean(x)
        d = self.residual_eigvals  # posterior covariance eigenvalues
        V = self.residual_eigvecs

        a = gamma/(1+gamma)

        # Likelihood precision eigenvalues: max(1/d - 1, 0)
        # Truncate at 0 for directions where posterior is wider than prior
        lambda_like = (1.0 / d - 1.0).clamp(min=0)

        # Proposal precision eigenvalues: a * lambda_like + 1
        lambda_prop = a * lambda_like + 1.0

        # Proposal variance eigenvalues
        var_prop = 1.0 / lambda_prop

        # Mean shrinkage in eigenbasis
        # alpha_i = a / (d_i * lambda_prop_i)
        mean_proj = V.T @ mean.T  # (ndim, batch)
        alpha = a / (d * lambda_prop)
        mean_prop = (V @ (alpha.unsqueeze(1) * mean_proj)).T  # (batch, ndim)

        # Sample: mean_prop + V @ diag(sqrt(var_prop)) @ eps
        eps = torch.randn_like(mean)
        return mean_prop + (V @ (torch.sqrt(var_prop).unsqueeze(1) * eps.T)).T


def sample_proposal(model, x_obs, n, gamma, device, step):
    """Sample from proposal = tempered_likelihood × prior."""
    with torch.no_grad():
        if step == 0:
            return sample_prior(n, device)
        x_batch = x_obs.expand(n, -1)
        return model.sample(x_batch, gamma=gamma)


def sample_posterior(model, x_obs, n, gamma, device):
    with torch.no_grad():
        x_batch = x_obs.expand(n, -1)
        gamma_eff = 1/gamma  # turn proposal sampler into posterior sampler mathematically
        return model.sample(x_batch, gamma=gamma)



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
        min_var=cfg.min_var,
        eig_update_freq=cfg.eig_update_freq,
    ).double().to(device)

    print(f"Using full covariance with eigenvalue-based tempered likelihood proposal")
    print(f"Eigendecomposition update frequency: {cfg.eig_update_freq}")
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
        loss = -model.log_prob(z_batch, x_batch).mean()
        loss.backward()
        optimizer.step()
        model.update_covariance(z_batch, x_batch)

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
        loss = -model.log_prob(z_batch, x_batch).mean()
        loss.backward()
        optimizer.step()
        model.update_covariance(z_batch, x_batch)

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
