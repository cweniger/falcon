# Embedding Notes for Linear Regression Posterior Learning

## Setup

Linear regression model: `y = Phi @ theta + noise`
- `Phi[i, k] = sin((k+1) * x_i)`, `x` on `[0, 2*pi)`
- `M=100` bins, `D=10` parameters, `sigma=0.1`
- Prior: `theta ~ N(0, I)`
- Analytic posterior available for validation

Five standalone scripts implement this with increasing modifications:
- `gaussian_lr.py` — full covariance (Cholesky) input/output normalization
- `gaussian_lr2.py` — diagonal normalization, `--zero_init` option
- `gaussian_lr3.py` — diagonal normalization + embedding network (`--embedding`)
- `gaussian_lr4.py` — simulation buffer with configurable replacement fraction
- `gaussian_lr5.py` — validated best-proposal mechanism (ratchet)

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

## fft_norm embedding: works at large n_bins

The `fft_norm` embedding works well even at very large `n_bins`:

```bash
python gaussian_lr3.py --n_bins 200000 --sigma_obs 10 --beta1 0.5 --beta2 0.5 --weight_decay 0.01 --lr2 0.001 --embedding fft_norm --n_modes 1280
```

### Why this works

The key idea: **use a fixed, well-conditioned transform (FFT) for the bulk of the
compression, then learn only a small gating layer on top.**

The signal in this problem is `Phi @ theta` where `Phi[i,k] = sin((k+1)*x_i)`,
so the signal lives entirely in frequency modes 1-10. The observation noise
spreads uniformly across all `n_bins/2+1 = 100001` frequency modes.

The `fft_norm` pipeline:
1. **Orthonormalized FFT** (`norm='ortho'`): coefficients are O(1) regardless
   of `n_bins`. This is a fixed, non-learned transform — no gradients needed
   for 200000-dimensional input weights.
2. **Frequency cutoff** (`--n_modes 1280`): keeps 1280 out of 100001 modes.
   Throws away ~99% of the noise while keeping all signal. The cutoff doesn't
   need to be tight — 1280 >> 10 signal modes — just small enough to make the
   learned layer tractable.
3. **Learned linear gating** (2×1280 = 2560 → `n_features`): the only learned
   part. Maps a manageable 2560-dim input to 20 features. Much better conditioned
   than learning a 200000→20 projection from scratch.

### Why learned linear embeddings fail at large n_bins

A `--embedding linear` layer at `n_bins=200000` has 200000 × n_features weights
that all need to receive useful gradient signal. The embedding never gets clear
gradients because:
- The signal is D=10-dimensional in a 200000-dimensional space
- Most weight directions correspond to pure noise
- The embedding gets confused by noise before it can lock onto the signal

The FFT sidesteps this entirely: the transform is fixed and known to concentrate
the signal into a small number of modes. The learning problem reduces from
"find 10 signal directions in R^200000" to "select from 1280 frequency bins".

### Continuum limit invariance

With `norm='ortho'`, the FFT coefficients are O(1) as n_bins → ∞. Combined with
`sigma_obs ∝ sqrt(n_bins)` scaling (which keeps Fisher information constant),
the `fft_norm` embedding output is stable across bin refinements. Fixed `--n_modes`
means the learned layer size doesn't change with `n_bins`.

## Simulation buffer and sample reuse (gaussian_lr4.py)

`gaussian_lr4.py` adds a fixed-size ring buffer with configurable replacement
fraction. Each training step draws a mini-batch from the buffer, then replaces
a fraction of the oldest entries with fresh proposal samples. This decouples
the number of forward simulations from the number of training steps.

Key issue discovered: **circular validation**. With heavy buffer reuse and a
small buffer, the model's EMA eigenvalues can collapse → narrow proposal →
validation samples from that proposal are tightly clustered → empirical
residual covariance is artificially small → `mean_std_r` drops below 1.0 even
though the model is wrong. The validation metric is contaminated by the
proposal distribution it's supposed to be evaluating.

## Validated best-proposal mechanism (gaussian_lr5.py)

`gaussian_lr5.py` solves the circular validation problem with a ratchet:

1. **Two copies of the model**: the training model (updated every step) and the
   best-proposal model (a `deepcopy`, updated only on improvement).
2. **Buffer samples come from the best model**, not the current training model.
3. **Validation draws fresh samples from the best model's proposal**, then
   evaluates the current training model on them.
4. **The best model is updated only when validation loss improves.**

This prevents collapse: a narrowing training model can't fool validation because
validation samples come from the (broader) previous-best proposal. The proposal
only advances when there's genuine improvement on the broader distribution.

### Baseline run

```bash
python gaussian_lr5.py --sigma_obs 1 --n_bins 20000 --embedding fft_norm --n_features 128 --n_modes 128 --replacement_fraction 0.5 --num_warmup 1 --buffer_size 4096 --beta1 0.1 --beta2 0.1 --gamma 0.1
```

Results:
- `mean_std_r` converges to ~1.02 and stays stable (no collapse)
- Proposal updated only 20 times across 10000 steps — most updates in the first
  ~1700 steps during rapid convergence, then the ratchet holds steady
- Mean RMSE vs analytic posterior: 0.0005 (excellent)
- Posterior std ratio: 0.91 (slight underestimate, expected with gamma=0.1)
- Total simulations: ~640k, wall time: ~39s on GPU

The key observation: `mean_std_r` now reflects actual posterior quality, not
proposal narrowness. With `gaussian_lr4.py` the same settings showed
`mean_std_r` dropping below 0.5 due to circular validation. The ratchet
mechanism eliminates this failure mode entirely.

### Flags

- `--val_every 100`: steps between validation checks
- `--val_samples 512`: fresh samples per validation
- `--patience 0`: if >0, revert training model to best after this many
  non-improving validations (marked `R` in output)

## Open questions

- Does SGD avoid the n_features sensitivity for learned linear embeddings?
- Would full covariance (Cholesky) normalization help independently of the
  optimizer, by making the loss landscape better conditioned?
- Can `n_modes` be set automatically from the eigenvalue spectrum of the FFT
  coefficients (analogous to PCA variance thresholding)?
