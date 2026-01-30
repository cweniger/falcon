# Explosion Mechanism in Sequential NPE with Gaussian Posteriors

This document summarizes the investigation into why training diverges around step 5000 when learning a Gaussian posterior with MSE loss and separately-estimated variance.

## Setup

- **Problem**: Learn posterior for `z` given `x = z + noise`, where `noise ~ N(0, σ_obs²)`
- **True posterior**: `N(x_obs, σ_obs²)`
- **Model**: MLP predicting mean, with variance estimated via EMA of residuals
- **Training**: MSE loss for mean, separate EMA update for log_var
- **Proposal**: Sample from `N(mean_pred, var_learned × temperature)` with temperature = 2

## Observation

Training converges to std_ratio ≈ 1.0 by step ~1000, stays stable until step ~5000, then explodes.

## Root Cause: Bias Crossing σ_obs Threshold

The model learns an identity function with a small bias:
```
model(x) ≈ x + bias, where bias = mean_pred - x_obs
```

The batch variance depends on this bias:
```
z - model(x) = z - x - bias = -noise - bias
batch_var = σ_obs² + bias²
```

**Critical threshold**: When `|bias| > σ_obs`:
- `batch_var ≈ bias²` (bias term dominates)
- System transitions from stable "identity regime" to unstable "constant regime"

## The Two Regimes

### Identity Regime (stable): |bias| << σ_obs
- `batch_var ≈ σ_obs²` (independent of proposal)
- Variance estimate correctly tracks observation noise
- Stable fixed point at std_ratio = 1

### Constant Regime (unstable): |bias| >> σ_obs
- `batch_var ≈ bias² ∝ var_proposal`
- Variance estimate depends on proposal width
- Positive feedback loop → explosion

## Why the Model Develops Bias

1. **Covariate shift**: Training samples cluster around `mean_pred`, not `x_obs`
2. **Biased gradient**: Samples near `x_obs` are drawn from `N(mean_pred, ...)`, so their expected value is `mean_pred`, not `x_obs`
3. **No corrective signal**: Once `mean_pred ≠ x_obs`, training reinforces the drift rather than correcting it

## Why It Takes ~5000 Steps

1. **Slow random walk**: `mean_pred` fluctuates around `x_obs` with small random errors
2. **Weak positive feedback**: Each step slightly amplifies the bias
3. **Threshold crossing**: Around step 5000, accumulated bias exceeds σ_obs
4. **Explosive feedback**: Once in "constant regime", bias grows exponentially

## Key Insight: Decoupled Loss

The instability arises because:
- **MSE loss** trains the model (prefers identity when var_proposal > σ_obs²)
- **EMA** updates variance separately (measures residuals from whatever model learned)

If the loss included log_var (like NLL), the dynamics would be coupled and might find different equilibria. The decoupling creates a situation where:
- Identity is stable (batch_var = σ_obs²)
- Constant is unstable (batch_var = var_proposal → runaway)
- Biased identity transitions from stable to unstable as |bias| crosses σ_obs

## Why Re-convergence Fails

During initial convergence (warmup with prior samples):
- Wide distribution covers `x_obs`
- Model learns `model(x) = x` globally
- Unbiased training at `x_obs`

During divergence:
- Narrow proposal centered at `mean_pred ≠ x_obs`
- Samples near `x_obs` are biased toward `mean_pred`
- Gradient reinforces drift, doesn't correct it

## Investigation: Why Importance Weighting (IW) Doesn't Help

We tested IW with weights `w = prior(z) / proposal(z)`. Result: **Still explodes!**

### The Problem

When the proposal drifts (mean_pred ≠ x_obs), all samples are ~1000σ away from the true posterior center. The prior is essentially **flat** in this narrow region:
- `log p(0.301) - log p(0.300) ≈ 3e-4` (negligible!)
- IW just reweights by `1/proposal(z)`, giving higher weight to tails
- But this doesn't shift the weighted mean toward x_obs

**Key insight**: IW can only reweight samples we have. It can't create samples where we have none.

### Smaller γ Makes It Worse

Testing with smaller γ (wider proposal):
- γ=0.5: Still explodes
- γ=0.1: Sometimes recovers (seed-dependent)
- γ=0.05: Catastrophic explosion (ESS ≈ 1 due to extreme weight variance)

The prior/proposal weight variance grows with proposal width, causing ESS collapse.

## Investigation: Neural Likelihood Estimation (NLE)

Since NPE suffers from covariate shift, we tested NLE (learning p(x|z) instead of p(z|x)). The likelihood is formally proposal-independent.

### Result: Still Fails (Differently)

NLE with 100% proposal sampling:
- **Mean**: Identity degrades over time (mu(0) = 0.03, mu(1) = 0.93)
- **Variance**: Doesn't converge (std_ratio = 3500)

**Why**: When training only in a narrow z region, the network "forgets" the global identity relationship.

## Solution: Mixed Sampling

The fix is simple: **always include some prior samples** in each batch.

| Prior Fraction | Mean Error | std_ratio | Stability |
|---------------|------------|-----------|-----------|
| 0% (default)  | 2.08e-4    | 2371      | ❌ Explodes |
| 20%           | 3.67e-4    | 1135      | ✓ Stable  |
| 50%           | 4.79e-4    | 1574      | ✓ Stable  |
| 80%           | 2.58e-3    | 1105      | ✓ Stable  |

### Why Mixed Sampling Works

- Prior samples maintain the global identity relationship
- Prevents complete covariate shift
- The network sees diverse z values, not just the narrow proposal region

### The Remaining Problem: Variance

Even with mixed sampling, **std_ratio never converges to 1.0**. This is fundamental:

The learned variance = E[(z - model(x))²] includes:
1. True noise: σ_obs² = 10⁻¹²
2. Network approximation error: E[e(x)²] ∼ 10⁻⁶

When σ_obs is extremely small, any network error dominates the variance estimate.

**Proof**: With a "pure identity" model (mu(z) = z exactly, only learn variance), std_ratio converges to 1.0 perfectly!

## Implications for Precision Cosmology

For very narrow posteriors (σ_obs ~ 10⁻⁶):

1. **Mean can be learned correctly** with mixed sampling
2. **Variance will be overestimated** due to irreducible network error
3. **This is conservative** - wider posteriors are safer than too narrow

### Possible Solutions

1. **Known parametric forms**: If x = f(z) is known, encode it directly
2. **Residual architectures**: Learn mu(z) = z + small_residual(z)
3. **Variance calibration**: Separately calibrate on held-out data
4. **Accept the limitation**: NN-based SBI has intrinsic resolution limits

### Falcon-Specific Recommendations

The buffer system (VALIDATION → TRAINING → DISFAVOURED → TOMBSTONE) acts like mixed sampling. To ensure stability:
1. Keep samples from wider distributions in the buffer
2. Consider explicit prior mixing parameter in training
3. Don't expect variance to converge for extremely narrow posteriors

## Files

- `explosion_example.py`: Demonstrates the bias dynamics and threshold crossing
- `gaussian_test.py`: Test script with `--prior_fraction` and `--iw` options
