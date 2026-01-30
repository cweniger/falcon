#!/usr/bin/env python3
"""
Linear regression posterior learning with eigenvalue-based tempered likelihood proposal.
Full covariance version, following the structure of gaussian_full_cov.py.

Model: y = Phi @ theta + noise
  - Phi[i, k] = sin((k+1) * x_i), x_i on [0, 2*pi), M bins, D=10 parameters
  - Prior: theta ~ N(0, I)
  - Noise: N(0, sigma^2 * I), sigma = 0.1

Analytic posterior:
  Sigma_post = (Phi^T Phi / sigma^2 + I)^{-1}
  mu_post    = Sigma_post @ Phi^T @ y / sigma^2
"""

import argparse
import time
import torch
import torch.nn as nn
import numpy as np


# Number of parameters (fixed)
D = 10


def get_config():
    parser = argparse.ArgumentParser()
    # Problem setup
    parser.add_argument('--n_bins', type=int, default=100,
                        help='Number of observation bins (M)')
    parser.add_argument('--sigma_obs', type=float, default=0.1,
                        help='Observation noise std')
    # Training
    parser.add_argument('--num_steps', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_warmup', type=int, default=1000)
    parser.add_argument('--gamma', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--print_every', type=int, default=100)
    parser.add_argument('--prior_only', action='store_true',
                        help='Always use prior samples (no proposal sampling)')
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
    return cfg


# =============================================================================
# Forward Model: y = Phi @ theta + noise
# =============================================================================

def design_matrix(n_bins):
    """Build design matrix Phi[i, k] = sin((k+1) * x_i)."""
    x = np.linspace(0, 2 * np.pi, n_bins, endpoint=False)
    Phi = np.column_stack([np.sin((k + 1) * x) for k in range(D)])
    return Phi, x


# Cached design matrix (populated in main)
_PHI_NP = None
_PHI_TORCH = None


def init_design_matrix(n_bins):
    """Initialize the cached design matrix for a given n_bins."""
    global _PHI_NP, _PHI_TORCH
    _PHI_NP, _ = design_matrix(n_bins)
    _PHI_TORCH = None  # reset torch cache


def get_phi(device, dtype):
    """Get design matrix as torch tensor on the right device."""
    global _PHI_TORCH
    if _PHI_TORCH is None or _PHI_TORCH.device != device or _PHI_TORCH.dtype != dtype:
        _PHI_TORCH = torch.tensor(_PHI_NP, device=device, dtype=dtype)
    return _PHI_TORCH


def forward_model(theta, device=None, dtype=None):
    """y = Phi @ theta (no noise)."""
    if device is None:
        device = theta.device
    if dtype is None:
        dtype = theta.dtype
    Phi = get_phi(device, dtype)
    return theta @ Phi.T  # (batch, M)


def analytic_posterior(y_obs_np, sigma_obs):
    """Compute analytic posterior mean and covariance.

    Sigma_post = (Phi^T Phi / sigma^2 + I)^{-1}
    mu_post    = Sigma_post @ Phi^T @ y / sigma^2
    """
    PhiTPhi = _PHI_NP.T @ _PHI_NP
    precision = PhiTPhi / sigma_obs**2 + np.eye(D)
    Sigma_post = np.linalg.inv(precision)
    mu_post = Sigma_post @ (_PHI_NP.T @ y_obs_np / sigma_obs**2)
    return mu_post, Sigma_post


def simulate(theta, sigma_obs, n_bins):
    """Simulate y = Phi @ theta + noise."""
    y_clean = forward_model(theta)
    noise = torch.randn(theta.shape[0], n_bins, device=theta.device, dtype=theta.dtype) * sigma_obs
    return y_clean + noise


def sample_prior(n, device, dtype=torch.float64):
    """Sample from N(0, I) prior over theta."""
    return torch.randn(n, D, device=device, dtype=dtype)


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

    def __init__(self, obs_dim, hidden_dim=128, num_layers=3, activation='relu',
                 momentum=0.01, min_var=1e-20, eig_update_freq=1):
        super().__init__()
        self.ndim = D
        self.obs_dim = obs_dim
        self.net = build_mlp(obs_dim, hidden_dim, D, num_layers, activation)
        self.momentum = momentum
        self.min_var = min_var
        self.eig_update_freq = eig_update_freq
        self.step_counter = 0

        # Mean vectors
        self.register_buffer('input_mean', torch.zeros(obs_dim))
        self.register_buffer('output_mean', torch.zeros(D))

        # Covariance matrices
        self.register_buffer('input_cov', torch.eye(obs_dim))
        self.register_buffer('output_cov', torch.eye(D))
        self.register_buffer('residual_cov', torch.eye(D))

        # Cholesky for input/output normalization
        self.register_buffer('input_cov_chol', torch.eye(obs_dim))
        self.register_buffer('output_cov_chol', torch.eye(D))

        # Eigendecomposition of residual covariance
        self.register_buffer('residual_eigvals', torch.ones(D))
        self.register_buffer('residual_eigvecs', torch.eye(D))

    def _compute_cov(self, data, mean, dim):
        """Compute covariance matrix from data."""
        centered = data - mean
        n = data.shape[0]
        cov = (centered.T @ centered) / max(n - 1, 1)
        cov = cov + self.min_var * torch.eye(dim, device=data.device, dtype=data.dtype)
        return cov

    def _safe_cholesky(self, cov, dim):
        """Compute Cholesky decomposition with fallback."""
        try:
            return torch.linalg.cholesky(cov)
        except RuntimeError:
            cov = cov + 1e-4 * torch.eye(dim, device=cov.device, dtype=cov.dtype)
            return torch.linalg.cholesky(cov)

    def _update_eigendecomp(self):
        """Update eigendecomposition of residual covariance."""
        eigvals, eigvecs = torch.linalg.eigh(self.residual_cov)
        eigvals = eigvals.clamp(min=self.min_var)
        self.residual_eigvals.copy_(eigvals)
        self.residual_eigvecs.copy_(eigvecs)

    def update_stats(self, z, x):
        """Update running statistics. z=theta (D-dim), x=y (obs_dim-dim)."""
        with torch.no_grad():
            self.input_mean.lerp_(x.mean(dim=0), self.momentum)
            self.output_mean.lerp_(z.mean(dim=0), self.momentum)

            batch_input_cov = self._compute_cov(x, self.input_mean, self.obs_dim)
            batch_output_cov = self._compute_cov(z, self.output_mean, self.ndim)
            self.input_cov.lerp_(batch_input_cov, self.momentum)
            self.output_cov.lerp_(batch_output_cov, self.momentum)
            self.input_cov_chol.copy_(self._safe_cholesky(self.input_cov, self.obs_dim))
            self.output_cov_chol.copy_(self._safe_cholesky(self.output_cov, self.ndim))

    def update_covariance(self, z, x):
        """Update residual covariance and eigendecomposition."""
        with torch.no_grad():
            mean = self.forward_mean(x)
            residuals = z - mean

            batch_cov = self._compute_cov(residuals, torch.zeros(self.ndim, device=z.device, dtype=z.dtype), self.ndim)
            self.residual_cov.lerp_(batch_cov, self.momentum)

            self.step_counter += 1
            if self.step_counter % self.eig_update_freq == 0:
                self._update_eigendecomp()

    def forward_mean(self, x):
        """Predict mean using Cholesky-based whitening."""
        centered = (x - self.input_mean).T  # (obs_dim, batch)
        x_white = torch.linalg.solve_triangular(
            self.input_cov_chol, centered, upper=False
        ).T  # (batch, obs_dim)
        r = self.net(x_white)  # (batch, D)
        return self.output_mean + (self.output_cov_chol @ r.T).T  # (batch, D)

    def get_covariance(self):
        """Get the full DxD covariance matrix."""
        return self.residual_cov.clone()

    def log_prob(self, z, x):
        """Gaussian log probability using eigendecomposition."""
        mean = self.forward_mean(x)
        residuals = z - mean

        log_det = torch.log(self.residual_eigvals).sum()

        V = self.residual_eigvecs
        d = self.residual_eigvals
        r_proj = V.T @ residuals.T  # (D, batch)
        mahal = (r_proj ** 2 / d.unsqueeze(1)).sum(dim=0)  # (batch,)

        return -0.5 * (self.ndim * np.log(2 * np.pi) + log_det.detach() + mahal)

    def sample(self, x, gamma=1.0):
        """Sample from proposal = tempered_likelihood x prior."""
        mean = self.forward_mean(x)
        d = self.residual_eigvals
        V = self.residual_eigvecs

        a = gamma / (1 + gamma)

        lambda_like = (1.0 / d - 1.0).clamp(min=0)
        lambda_prop = a * lambda_like + 1.0
        var_prop = 1.0 / lambda_prop

        mean_proj = V.T @ mean.T  # (D, batch)
        alpha = a / (d * lambda_prop)
        mean_prop = (V @ (alpha.unsqueeze(1) * mean_proj)).T  # (batch, D)

        eps = torch.randn_like(mean)
        return mean_prop + (V @ (torch.sqrt(var_prop).unsqueeze(1) * eps.T)).T


def sample_proposal(model, y_obs, n, gamma, device, step, prior_only=False):
    """Sample from proposal = tempered_likelihood x prior."""
    with torch.no_grad():
        if prior_only or step == 0:
            return sample_prior(n, device)
        y_batch = y_obs.expand(n, -1)
        return model.sample(y_batch, gamma=gamma)


def sample_posterior(model, y_obs, n, gamma, device):
    with torch.no_grad():
        y_batch = y_obs.expand(n, -1)
        gamma_eff = 1 / gamma
        return model.sample(y_batch, gamma=gamma_eff)


# =============================================================================
# Main
# =============================================================================

def main():
    cfg = get_config()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.set_default_dtype(torch.float64)
    device = torch.device(cfg.device if torch.cuda.is_available() else 'cpu')

    n_bins = cfg.n_bins
    init_design_matrix(n_bins)

    # Generate true parameters from prior
    theta_true = np.random.randn(D)

    # Asimov observation (no noise for clean ground truth)
    y_obs_np = _PHI_NP @ theta_true

    # Compute analytic posterior
    mu_post, Sigma_post = analytic_posterior(y_obs_np, cfg.sigma_obs)
    marginal_std = np.sqrt(np.diag(Sigma_post))

    print(f"Device: {device}")
    print(f"Forward model: y = Phi @ theta + noise")
    print(f"  M={n_bins} bins, D={D} parameters, sigma={cfg.sigma_obs}")
    print(f"  Phi[i,k] = sin((k+1) * x_i), x in [0, 2*pi)")
    print()
    print(f"{'Param':<10} {'True':>8} {'Post Mean':>10} {'Post Std':>10}")
    print("-" * 40)
    for k in range(D):
        print(f"theta[{k}]  {theta_true[k]:>8.4f} {mu_post[k]:>10.4f} {marginal_std[k]:>10.6f}")
    print()
    print(f"Training: {cfg.num_steps} steps x {cfg.batch_size} = {cfg.num_steps * cfg.batch_size} samples")
    print()

    # Create model
    model = NormalizedMLP(
        obs_dim=n_bins,
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

    y_obs = torch.tensor(y_obs_np, device=device, dtype=torch.float64).unsqueeze(0)  # (1, M)

    # Print header
    print(f"{'Step':>6} {'Samples':>10} {'Loss':>12} {'mean_var_r':>12} {'mean_std_r':>12}")
    print("-" * 56)

    t0 = time.perf_counter()

    # Warmup
    print(f"[Warmup: {cfg.num_warmup} steps with prior samples]")
    for step in range(cfg.num_warmup):
        z_batch = sample_prior(cfg.batch_size, device)
        x_batch = simulate(z_batch, cfg.sigma_obs, n_bins)
        model.update_stats(z_batch, x_batch)
        optimizer.zero_grad()
        loss = -model.log_prob(z_batch, x_batch).mean()
        loss.backward()
        optimizer.step()
        model.update_covariance(z_batch, x_batch)

    mean_pred = model.forward_mean(y_obs)
    print(f"[Warmup done, mean(y_obs) first 3: {mean_pred.squeeze()[:3].tolist()}]")

    for pg in optimizer.param_groups:
        pg['lr'] = cfg.lr2
    print(f"[LR changed to {cfg.lr2} for sequential phase]\n")

    # Sequential phase
    for step in range(cfg.num_steps):
        z_batch = sample_proposal(model, y_obs, cfg.batch_size, cfg.gamma, device, step, cfg.prior_only)
        x_batch = simulate(z_batch, cfg.sigma_obs, n_bins)

        model.update_stats(z_batch, x_batch)
        optimizer.zero_grad()
        loss = -model.log_prob(z_batch, x_batch).mean()
        loss.backward()
        optimizer.step()
        model.update_covariance(z_batch, x_batch)

        if step % cfg.print_every == 0 or step == cfg.num_steps - 1:
            cov_pred = model.get_covariance().detach().cpu().numpy()
            diag_pred = np.diag(cov_pred)
            diag_expected = np.diag(Sigma_post)
            mean_var_ratio = np.mean(diag_pred / diag_expected)
            mean_std_ratio = np.mean(np.sqrt(diag_pred) / marginal_std)
            samples = (step + 1) * cfg.batch_size

            print(f"{step:>6} {samples:>10} {loss.item():>12.2e} {mean_var_ratio:>12.3f} {mean_std_ratio:>12.3f}")

    elapsed = time.perf_counter() - t0

    # Final results
    mean_pred = model.forward_mean(y_obs).squeeze().detach().cpu().numpy()
    cov_pred = model.get_covariance().detach().cpu().numpy()

    print()
    print("Final results (per-parameter):")
    print(f"{'Param':<10} {'True':>8} {'Analytic':>10} {'Learned':>10} {'Anal Std':>10} {'Learn Std':>10} {'Var Ratio':>10}")
    print("-" * 70)
    for k in range(D):
        learned_std = np.sqrt(cov_pred[k, k])
        var_ratio = cov_pred[k, k] / Sigma_post[k, k]
        print(f"theta[{k}]  {theta_true[k]:>8.4f} {mu_post[k]:>10.4f} {mean_pred[k]:>10.4f} "
              f"{marginal_std[k]:>10.6f} {learned_std:>10.6f} {var_ratio:>10.3f}")

    # Summary statistics
    mean_error = np.sqrt(np.mean((mean_pred - mu_post)**2))
    std_ratio = np.sqrt(np.diag(cov_pred)) / marginal_std
    print()
    print(f"  RMSE(mean_learned - mean_analytic): {mean_error:.6f}")
    print(f"  Std ratio (learned/analytic): {std_ratio.mean():.4f} +/- {std_ratio.std():.4f}")

    # Sample from posterior and compute statistics
    print("\nPosterior sampling:")
    n_posterior = 10000
    z_posterior = sample_posterior(model, y_obs, n_posterior, cfg.gamma, device)
    z_posterior_np = z_posterior.cpu().numpy()

    sample_mean = z_posterior_np.mean(axis=0)
    sample_cov = np.cov(z_posterior_np.T)
    sample_std = np.sqrt(np.diag(sample_cov))

    print(f"{'Param':<10} {'True':>8} {'Analytic':>10} {'Sample':>10} {'Anal Std':>10} {'Samp Std':>10} {'Std Ratio':>10}")
    print("-" * 70)
    for k in range(D):
        sr = sample_std[k] / marginal_std[k]
        print(f"theta[{k}]  {theta_true[k]:>8.4f} {mu_post[k]:>10.4f} {sample_mean[k]:>10.4f} "
              f"{marginal_std[k]:>10.6f} {sample_std[k]:>10.6f} {sr:>10.3f}")

    sample_mean_error = np.sqrt(np.mean((sample_mean - mu_post)**2))
    sample_std_ratio = sample_std / marginal_std
    print()
    print(f"  RMSE(sample_mean - analytic_mean): {sample_mean_error:.6f}")
    print(f"  Std ratio (sample/analytic): {sample_std_ratio.mean():.4f} +/- {sample_std_ratio.std():.4f}")

    print(f"\nTime: {elapsed:.2f}s")


if __name__ == "__main__":
    main()
