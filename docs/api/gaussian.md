# Gaussian Estimator

Full covariance Gaussian posterior estimation.

## Overview

`Gaussian` provides a simpler alternative to [Flow](flow.md) for posterior
estimation. Instead of normalizing flows, it models the posterior as a
multivariate Gaussian with full covariance, using eigenvalue-based operations.

Key features:

- Full covariance matrix showing parameter correlations directly
- Eigenvalue-based tempered sampling for exploration
- Simpler and more interpretable than flow-based methods

!!! note
    `Gaussian` requires a [`Product`](product.md) prior (with `"standard_normal"` mode)
    as the simulator, not [`Hypercube`](hypercube.md).

## Configuration

Gaussian is configured through the same four groups as Flow: `loop`, `network`,
`optimizer`, and `inference`.

```yaml
estimator:
  _target_: falcon.estimators.Gaussian
  loop:
    num_epochs: 1000
    batch_size: 128
    early_stop_patience: 32
  network:
    hidden_dim: 128
    num_layers: 3
    momentum: 0.10
    min_var: 1.0e-20
    eig_update_freq: 1
  embedding:
    _target_: model.E_identity
    _input_: [x]
  optimizer:
    lr: 0.01
    lr_decay_factor: 0.5
    scheduler_patience: 16
  inference:
    gamma: 1.0
    discard_samples: false
    log_ratio_threshold: -20.0
```

## Configuration Reference

### Network (`network`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `hidden_dim` | int | 128 | MLP hidden layer dimension |
| `num_layers` | int | 3 | Number of hidden layers |
| `momentum` | float | 0.10 | EMA momentum for running statistics |
| `min_var` | float | 1e-20 | Minimum variance for numerical stability |
| `eig_update_freq` | int | 1 | Eigendecomposition update frequency |

The `loop`, `optimizer`, and `inference` groups share the same parameters as
[Flow](flow.md#training-loop-loop).

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
      _target_: falcon.estimators.Gaussian
      loop:
        num_epochs: 1000
        batch_size: 128
        early_stop_patience: 32
      network:
        hidden_dim: 128
        num_layers: 3
        momentum: 0.10
        min_var: 1.0e-20
        eig_update_freq: 1
      embedding:
        _target_: model.E_identity
        _input_: [x]
      optimizer:
        lr: 0.01
        lr_decay_factor: 0.5
        scheduler_patience: 16
      inference:
        gamma: 1.0
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

::: falcon.estimators.gaussian.Gaussian

::: falcon.estimators.gaussian.GaussianPosterior
    options:
      show_source: true

## Configuration Classes

::: falcon.estimators.gaussian.GaussianConfig

::: falcon.estimators.gaussian.GaussianPosteriorConfig
