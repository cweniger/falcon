# Gaussian Estimator

Full covariance Gaussian posterior estimation.

## Overview

`GaussianFullCov` provides a simpler alternative to [Flow](flow.md) for posterior
estimation. Instead of normalizing flows, it models the posterior as a
multivariate Gaussian with full covariance, using eigenvalue-based operations.

Key features:

- Full covariance matrix showing parameter correlations directly
- Eigenvalue-based tempered sampling for exploration
- Simpler and more interpretable than flow-based methods

!!! note
    `GaussianFullCov` requires a [`Product`](product.md) prior with `"standard_normal"` mode.

## Configuration

Like `Flow`, all `GaussianFullCov` parameters are specified **flat** directly
under `estimator:` in YAML — no nested group keys.

```yaml
estimator:
  _target_: falcon.estimators.GaussianFullCov
  max_epochs: 1000
  batch_size: 128
  early_stop_patience: 32
  hidden_dim: 128
  num_layers: 3
  momentum: 0.01
  min_var: 1.0e-20
  eig_update_freq: 1
  embedding:
    _target_: model.E_identity
    _input_: [x]
  lr: 0.01
  lr_decay_factor: 1.0
  lr_patience: 8
  gamma: 0.5
  discard_samples: false
  log_ratio_threshold: -20.0
```

## Configuration Reference

### Network Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `hidden_dim` | int | 128 | MLP hidden layer dimension |
| `num_layers` | int | 3 | Number of hidden layers |
| `momentum` | float | 0.01 | EMA momentum for running statistics |
| `min_var` | float | 1e-20 | Minimum variance for numerical stability |
| `eig_update_freq` | int | 1 | Eigendecomposition update frequency |

The training loop, optimizer, and inference parameters (`max_epochs`, `batch_size`,
`early_stop_patience`, `lr`, `lr_decay_factor`, `lr_patience`, `prior_epochs`,
`gamma`, `discard_samples`, `log_ratio_threshold`, etc.) are identical to those in
[Flow](flow.md#configuration-reference).

!!! note "gamma for GaussianFullCov"
    Unlike `Flow`, where `gamma` controls importance-sampling breadth, in
    `GaussianFullCov` it controls eigenvalue tempering of the covariance matrix.
    Smaller `gamma` (e.g. `0.1`) produces a broader proposal relative to the
    posterior; the relationship is not the same as in `Flow`.

## Complete Example

```yaml
graph:
  z:
    evidence: [x]

    simulator:
      _target_: falcon.priors.Product
      priors:
        - ['normal', 0.0, 1.0]
        - ['normal', 0.0, 1.0]
        - ['normal', 0.0, 1.0]

    estimator:
      _target_: falcon.estimators.GaussianFullCov
      max_epochs: 8000
      batch_size: 128
      early_stop_patience: 128
      hidden_dim: 128
      num_layers: 3
      momentum: 0.01
      min_var: 1.0e-20
      eig_update_freq: 1
      embedding:
        _target_: model.E_identity
        _input_: [x]
      lr: 0.01
      lr_decay_factor: 1.0
      lr_patience: 8
      gamma: 0.1
      discard_samples: false
      log_ratio_threshold: -20.0

    ray:
      num_gpus: 1

  x:
    parents: [z]
    simulator:
      _target_: model.ExpPlusNoise
      sigma: 1.0e-6
    observed: "./data/mock_data.npz['x']"

sample:
  posterior:
    n: 1000
```

## Class Reference

::: falcon.estimators.gaussian_fullcov.GaussianFullCov
    options:
      show_source: true
      members:
        - __init__
        - train_step
        - val_step
        - sample_prior
        - sample_posterior
        - sample_proposal
        - save
        - load
