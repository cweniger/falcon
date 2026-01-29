#!/usr/bin/env python3
"""Generate mock data for the linear regression example.

Model: y = Phi @ theta + noise
  - Phi[i, k] = sin((k+1) * x_i), 100 bins, 10 parameters
  - Prior: theta ~ N(0, I)
  - Noise: N(0, sigma^2 * I)

Also computes the analytic posterior for comparison.

For a linear model with Gaussian prior and likelihood:
  Prior:      theta ~ N(0, I)
  Likelihood: y | theta ~ N(Phi @ theta, sigma^2 * I)
  Posterior:  theta | y ~ N(mu_post, Sigma_post)

  Sigma_post = (Phi^T Phi / sigma^2 + I)^{-1}
  mu_post    = Sigma_post @ Phi^T @ y / sigma^2

Convention: Observations have no batch dimension - shape is [n_bins].
"""

import numpy as np
import sys
sys.path.insert(0, "../src")
from model import design_matrix

# Configuration
N_BINS = 100
N_PARAMS = 10
SIGMA = 0.1

# True parameters (drawn from prior for a realistic test)
np.random.seed(42)
theta_true = np.random.randn(N_PARAMS)

# Design matrix
Phi, x = design_matrix(N_BINS, N_PARAMS)

# Observation: y = Phi @ theta (Asimov, no noise for clean testing)
y_obs = Phi @ theta_true

# Analytic posterior
# Sigma_post = (Phi^T Phi / sigma^2 + I)^{-1}
PhiTPhi = Phi.T @ Phi
precision_post = PhiTPhi / SIGMA**2 + np.eye(N_PARAMS)
Sigma_post = np.linalg.inv(precision_post)

# mu_post = Sigma_post @ Phi^T @ y / sigma^2
mu_post = Sigma_post @ (Phi.T @ y_obs / SIGMA**2)

# Marginal posterior widths
marginal_std = np.sqrt(np.diag(Sigma_post))

# Print results
print(f"Linear regression: y = Phi @ theta + noise")
print(f"  {N_PARAMS} parameters, {N_BINS} bins, sigma = {SIGMA}")
print(f"  Phi[i, k] = sin((k+1) * x_i), x in [0, 2*pi)")
print()

print(f"{'Param':<8} {'True':>10} {'Post Mean':>10} {'Post Std':>10}")
print("-" * 40)
for k in range(N_PARAMS):
    print(f"theta[{k}] {theta_true[k]:>10.4f} {mu_post[k]:>10.4f} {marginal_std[k]:>10.6f}")

print(f"\nPosterior correlation matrix (off-diagonal elements):")
corr = Sigma_post / np.outer(marginal_std, marginal_std)
print(np.array2string(corr, precision=3, suppress_small=True))

# Save
np.savez("mock_data.npz",
         y=y_obs,  # observation
         theta_true=theta_true,
         mu_post=mu_post,
         Sigma_post=Sigma_post,
         marginal_std=marginal_std)
print(f"\nSaved to mock_data.npz (y_obs shape: {y_obs.shape})")
