#!/usr/bin/env python3
"""
Linear regression posterior learning with eigenvalue-based tempered likelihood proposal.
Diagonal normalization version with optional embedding network.

Model: y = Phi @ theta + noise
  - Phi[i, k] = sin((k+1) * x_i), x_i on [0, 2*pi), M bins, D=10 parameters
  - Prior: theta ~ N(0, I)
  - Noise: N(0, sigma^2 * I), sigma = 0.1

Embedding modes (--embedding):
  none    - no embedding, MLP takes raw n_bins input
  linear  - single linear layer: n_bins -> n_features
  fft     - real FFT, then learned gating to n_features
  fft_norm - orthonormalized FFT with frequency cutoff, gated to n_features
  gated   - wide linear layer with learned sigmoid gates + L1 penalty

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
    parser.add_argument('--zero_init', action='store_true',
                        help='Initialize first MLP layer weights and bias to zero')
    parser.add_argument('--no_whiten', action='store_true',
                        help='Disable input/output diagonal whitening')
    # Embedding
    parser.add_argument('--embedding', type=str, default='none',
                        choices=['none', 'linear', 'fft', 'fft_norm', 'gated'],
                        help='Embedding mode: none, linear, fft, fft_norm, or gated')
    parser.add_argument('--n_features', type=int, default=20,
                        help='Embedding output dimension (input to MLP)')
    parser.add_argument('--n_modes', type=int, default=0,
                        help='fft_norm: number of frequency modes to keep (0 = all)')
    parser.add_argument('--gate_l1', type=float, default=0.01,
                        help='Gated: L1 penalty weight on gate activations')
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
# Embedding networks
# =============================================================================

class LinearEmbedding(nn.Module):
    """Single linear layer: n_bins -> n_features."""

    def __init__(self, n_bins, n_features):
        super().__init__()
        self.linear = nn.Linear(n_bins, n_features)

    def forward(self, x):
        return self.linear(x)


class FFTEmbedding(nn.Module):
    """Real FFT followed by a learned linear gating to n_features.

    Computes rfft, stacks real and imaginary parts, then applies a linear
    layer to select/combine frequency components down to n_features.
    """

    def __init__(self, n_bins, n_features):
        super().__init__()
        # rfft of n_bins real values gives n_bins//2 + 1 complex coefficients
        # stacking real and imaginary parts -> 2 * (n_bins//2 + 1) features
        n_fft = 2 * (n_bins // 2 + 1)
        self.gate = nn.Linear(n_fft, n_features)

    def forward(self, x):
        fft_c = torch.fft.rfft(x, dim=-1)
        fft_real = torch.cat([fft_c.real, fft_c.imag], dim=-1)
        return self.gate(fft_real)


class FFTNormEmbedding(nn.Module):
    """Orthonormalized FFT with frequency cutoff, gated to n_features.

    Uses norm='ortho' so coefficients are O(1) regardless of n_bins.
    Keeps only the first n_modes frequency modes (low-pass), stacks
    real and imaginary parts, then linearly maps to n_features.
    """

    def __init__(self, n_bins, n_features, n_modes=0):
        super().__init__()
        max_modes = n_bins // 2 + 1
        self.n_modes = min(n_modes, max_modes) if n_modes > 0 else max_modes
        n_fft = 2 * self.n_modes
        self.gate = nn.Linear(n_fft, n_features)

    def forward(self, x):
        fft_c = torch.fft.rfft(x, dim=-1, norm='ortho')
        fft_c = fft_c[..., :self.n_modes]
        fft_real = torch.cat([fft_c.real, fft_c.imag], dim=-1)
        return self.gate(fft_real)


class GatedEmbedding(nn.Module):
    """Wide linear layer with learned sigmoid gates.

    Projects n_bins -> n_features via a linear layer, then element-wise
    multiplies by sigmoid(gate_logits). Gates are initialized to zero
    (sigmoid(0) = 0.5). L1 penalty on gate activations encourages the
    network to shut off unused dimensions.
    """

    def __init__(self, n_bins, n_features):
        super().__init__()
        self.linear = nn.Linear(n_bins, n_features)
        self.gate_logits = nn.Parameter(torch.zeros(n_features))

    def forward(self, x):
        h = self.linear(x)
        gates = torch.sigmoid(self.gate_logits)
        return h * gates

    def gate_l1(self):
        """L1 penalty on gate activations (for adding to loss)."""
        return torch.sigmoid(self.gate_logits).sum()


def build_embedding(mode, n_bins, n_features, n_modes=0):
    """Build embedding network. Returns (module, output_dim).

    For PCA, output_dim is n_bins initially (updated after finalize).
    For others, output_dim is known at construction time.
    """
    if mode == 'none':
        return nn.Identity(), n_bins
    elif mode == 'linear':
        return LinearEmbedding(n_bins, n_features), n_features
    elif mode == 'fft':
        return FFTEmbedding(n_bins, n_features), n_features
    elif mode == 'fft_norm':
        return FFTNormEmbedding(n_bins, n_features, n_modes=n_modes), n_features
    elif mode == 'gated':
        return GatedEmbedding(n_bins, n_features), n_features
    else:
        raise ValueError(f"Unknown embedding mode: {mode}")


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
    """MLP with optional embedding, diagonal online normalization,
    and eigenvalue-based covariance tracking.

    Pipeline: x -> diagonal whitening -> embedding -> MLP -> diagonal de-whitening
    The diagonal normalization operates on the raw observation (n_bins),
    the embedding compresses to n_features, and the MLP maps n_features -> D.
    """

    def __init__(self, obs_dim, hidden_dim=128, num_layers=3, activation='relu',
                 momentum=0.01, min_var=1e-20, eig_update_freq=1, zero_init=False,
                 embedding_mode='none', n_features=20, no_whiten=False, n_modes=0):
        super().__init__()
        self.ndim = D
        self.obs_dim = obs_dim
        self.embedding_mode = embedding_mode
        self.no_whiten = no_whiten

        # Embedding: obs_dim -> emb_dim
        self.embedding, emb_dim = build_embedding(embedding_mode, obs_dim, n_features, n_modes=n_modes)

        # MLP: emb_dim -> D
        self.net = build_mlp(emb_dim, hidden_dim, D, num_layers, activation)
        if zero_init:
            first_linear = self.net[0]
            nn.init.zeros_(first_linear.weight)
            nn.init.zeros_(first_linear.bias)

        self.momentum = momentum
        self.min_var = min_var
        self.eig_update_freq = eig_update_freq
        self.step_counter = 0

        # Diagonal normalization: mean and std vectors (on raw obs)
        self.register_buffer('input_mean', torch.zeros(obs_dim))
        self.register_buffer('input_std', torch.ones(obs_dim))
        self.register_buffer('output_mean', torch.zeros(D))
        self.register_buffer('output_std', torch.ones(D))

        # Residual covariance (full, for the posterior)
        self.register_buffer('residual_cov', torch.eye(D))

        # Eigendecomposition of residual covariance
        self.register_buffer('residual_eigvals', torch.ones(D))
        self.register_buffer('residual_eigvecs', torch.eye(D))

    def _update_eigendecomp(self):
        """Update eigendecomposition of residual covariance."""
        eigvals, eigvecs = torch.linalg.eigh(self.residual_cov)
        eigvals = eigvals.clamp(min=self.min_var)
        self.residual_eigvals.copy_(eigvals)
        self.residual_eigvecs.copy_(eigvecs)

    def update_stats(self, z, x):
        """Update running mean and std. z=theta (D-dim), x=y (obs_dim-dim)."""
        with torch.no_grad():
            self.input_mean.lerp_(x.mean(dim=0), self.momentum)
            self.output_mean.lerp_(z.mean(dim=0), self.momentum)

            input_var = x.var(dim=0).clamp(min=self.min_var)
            output_var = z.var(dim=0).clamp(min=self.min_var)
            self.input_std.lerp_(input_var.sqrt(), self.momentum)
            self.output_std.lerp_(output_var.sqrt(), self.momentum)

    def update_covariance(self, z, x):
        """Update residual covariance and eigendecomposition."""
        with torch.no_grad():
            mean = self.forward_mean(x)
            residuals = z - mean

            n = residuals.shape[0]
            batch_cov = (residuals.T @ residuals) / max(n - 1, 1)
            batch_cov = batch_cov + self.min_var * torch.eye(self.ndim, device=z.device, dtype=z.dtype)
            self.residual_cov.lerp_(batch_cov, self.momentum)

            self.step_counter += 1
            if self.step_counter % self.eig_update_freq == 0:
                self._update_eigendecomp()

    def forward_mean(self, x):
        """Predict mean: whiten -> embed -> MLP -> de-whiten."""
        if self.no_whiten:
            h = self.embedding(x)
            return self.net(h)
        x_white = (x - self.input_mean) / self.input_std
        h = self.embedding(x_white)
        r = self.net(h)  # (batch, D)
        return self.output_mean + self.output_std * r

    def gate_l1_loss(self):
        """Return gate L1 penalty (0 if not using gated embedding)."""
        if self.embedding_mode == 'gated':
            return self.embedding.gate_l1()
        return 0.0

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
        zero_init=cfg.zero_init,
        embedding_mode=cfg.embedding,
        n_features=cfg.n_features,
        no_whiten=cfg.no_whiten,
        n_modes=cfg.n_modes,
    ).double().to(device)

    gate_l1 = cfg.gate_l1 if cfg.embedding == 'gated' else 0.0

    emb_desc = cfg.embedding if cfg.embedding != 'none' else 'none (raw input)'
    print(f"Embedding: {emb_desc}" + (f", n_features={cfg.n_features}" if cfg.embedding != 'none' else ''))
    if cfg.embedding == 'gated':
        print(f"  Gate L1 penalty: {gate_l1}")
    print(f"Diagonal normalization with eigenvalue-based tempered likelihood proposal")
    print(f"Eigendecomposition update frequency: {cfg.eig_update_freq}")
    print()

    optimizer = torch.optim.AdamW(
        model.parameters(),
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
        loss = -model.log_prob(z_batch, x_batch).mean() + gate_l1 * model.gate_l1_loss()
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
        loss = -model.log_prob(z_batch, x_batch).mean() + gate_l1 * model.gate_l1_loss()
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

    if cfg.embedding == 'gated':
        gates = torch.sigmoid(model.embedding.gate_logits).detach().cpu().numpy()
        active = (gates > 0.5).sum()
        print(f"\nGated embedding: {active}/{len(gates)} gates active (>0.5)")
        print(f"  Gate values (sorted): {np.sort(gates)[::-1].tolist()}")

    print(f"\nTime: {elapsed:.2f}s")


if __name__ == "__main__":
    main()
