# Embedding Notes for Linear Regression Posterior Learning

## Setup

Linear regression model: `y = Phi @ theta + noise`
- `Phi[i, k] = sin((k+1) * x_i)`, `x` on `[0, 2*pi)`
- `M=100` bins, `D=10` parameters, `sigma=0.1`
- Prior: `theta ~ N(0, I)`
- Analytic posterior available for validation

Three standalone scripts implement this with increasing modifications:
- `gaussian_lr.py` — full covariance (Cholesky) input/output normalization
- `gaussian_lr2.py` — diagonal normalization, `--zero_init` option
- `gaussian_lr3.py` — diagonal normalization + embedding network (`--embedding`)

## Key finding: linear embedding sensitivity to n_features

A single linear embedding layer before the MLP (`--embedding linear`) dramatically
improves convergence compared to no embedding — but only for `n_features` in the
range ~20-24. Fewer or more features give poor results.

### Why it's not redundant

At first glance, a linear embedding before the MLP's first (also linear) layer should
collapse to a single linear map. But the diagonal normalization between the raw input
and the embedding breaks this equivalence. The pipeline is:

```
x -> (x - mean) / std -> embedding(n_bins -> n_features) -> MLP(n_features -> D)
```

The diagonal whitening rotates the signal subspace away from axis alignment. Two
stacked linear layers with Adam/AdamW then differ from one in three ways:

1. **Implicit low-rank bottleneck.** The product `W_mlp @ W_emb` is rank-limited
   to `n_features`. For this problem the signal lives in a ~D-dimensional subspace
   of the M=100-dim observation. The bottleneck acts as a regularizer that forces
   the network to find the relevant subspace.

2. **Weight decay acts differently on factored matrices.** AdamW penalizes
   `||W_emb||^2 + ||W_mlp||^2`, which biases toward low nuclear norm of the
   effective matrix — different from penalizing `||W_eff||^2` directly.

3. **Adam's per-parameter learning rates.** Gradient second-moment statistics
   differ for `W_emb` and `W_mlp` vs. a single fused `W`, leading to different
   optimization trajectories.

### Why ~2D features

The sufficient statistic is `Phi^T @ y` (D=10 dimensional). After diagonal
whitening, the D signal directions are rotated and no longer axis-aligned. The
embedding needs headroom beyond exactly D to stably represent the rotated signal
subspace during training. Empirically, ~2D features (20-24) provides this headroom.

With fewer features: not enough capacity to capture the full signal subspace.
With more features: the bottleneck regularization disappears and the network
overfits to the narrow proposal distribution during sequential training.

### Prediction

With `--prior_only` (amortized training, no distribution shift), the sensitivity
to `n_features` should largely disappear — larger values should work because
there's no proposal-induced overfitting.

## Reproducing results

```bash
# Baseline: no embedding (struggles)
python gaussian_lr3.py --embedding none

# Linear embedding, sweet spot (works well)
python gaussian_lr3.py --embedding linear --n_features 20

# Tested with: n_features 20-24 work, more or less gives bad results.
# NOTE: results are very sensitive to these specific settings (beta1, beta2,
# weight_decay, lr2, sigma_obs, n_bins). Small changes can break convergence.
python gaussian_lr3.py --n_bins 10000 --sigma_obs 3 --embedding linear --beta1 0.5 --beta2 0.5 --weight_decay 0.01 --n_features 25 --lr2 0.001

# Linear embedding, too few features (poor)
python gaussian_lr3.py --embedding linear --n_features 10

# Linear embedding, too many features (poor)
python gaussian_lr3.py --embedding linear --n_features 50

# FFT embedding
python gaussian_lr3.py --embedding fft --n_features 20

# Test prediction: prior_only should be less sensitive to n_features
python gaussian_lr3.py --embedding linear --n_features 50 --prior_only

# Diagonal normalization without embedding (gaussian_lr2.py)
python gaussian_lr2.py
python gaussian_lr2.py --zero_init

# Full covariance normalization (gaussian_lr.py, original)
python gaussian_lr.py
python gaussian_lr.py --prior_only
```

## Gated embedding: also brittle

The gated embedding (`--embedding gated`) was implemented to automatically learn
the right effective rank. In practice, it is similarly brittle. The sigmoid gates
+ L1 penalty add their own hyperparameters (`--gate_l1`, `--n_features` width)
that need tuning, and getting the L1 strength right is just as fragile as picking
`n_features` directly.

The core issue is that diagonal normalization creates a coupling between the
whitening and the embedding — the effective signal subspace (after whitening) is
hard to learn robustly regardless of how the bottleneck is implemented (fixed rank,
learned gates, etc.).

## Whitening is not the cause

Disabling input/output whitening entirely (`--no_whiten`) shows the same
brittleness. The problem is present during warmup (prior samples only, no
sequential distribution shift) and with or without diagonal normalization.

This rules out:
- Diagonal normalization distorting the signal subspace
- Sequential proposal-induced overfitting

The likely root cause is **Adam's per-parameter learning rates** interacting with
the factored linear bottleneck (embedding W1 × MLP first layer W2). Adam's
second-moment normalization gives each weight its own effective learning rate,
creating asymmetric scaling between the two factors. This makes optimization
sensitive to the relative magnitudes, which change with `n_features`, `n_bins`,
and `sigma_obs`. SGD with a single global learning rate would not have this
issue (though it converges slower).

## Open questions

- Does SGD avoid the n_features sensitivity?
- Would full covariance (Cholesky) normalization help independently of the
  optimizer, by making the loss landscape better conditioned?
