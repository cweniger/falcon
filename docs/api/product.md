# Product Prior

Product of independent marginal distributions with latent space transformations.

## Overview

`Product` defines a prior as a product of independent 1D marginals, each with a
bijective map to a chosen latent space. It extends the abstract base class
`TransformedPrior`, which defines the `forward`/`inverse` interface.

`Product` supports two latent-space modes:

- **`"hypercube"`**: Maps to/from a bounded hypercube (used with [Flow](flow.md))
- **`"standard_normal"`**: Maps to/from standard normal space (used with [GaussianFullCov](gaussian.md))

## Supported Distributions

| Type | Parameters | Description |
|------|------------|-------------|
| `uniform` | `low`, `high` | Uniform distribution over [`low`, `high`] |
| `normal` | `mean`, `std` | Gaussian distribution |
| `cosine` | `low`, `high` | Distribution with pdf ∝ cos(θ) — use for inclination-like angles |
| `sine` | `low`, `high` | Distribution with pdf ∝ sin(θ) — use for declination-like angles |
| `uvol` | `low`, `high` | Uniform-in-volume (pdf ∝ r²) — use for a radial coordinate in 3D |
| `triangular` | `a`, `c`, `b` | Triangular distribution (min `a`, mode `c`, max `b`) |
| `lognormal` | `mean`, `std` | Log-normal distribution. Only supported with `"standard_normal"` mode; raises `ValueError` with `"hypercube"` mode |
| `fixed` | `value` | Fixed (non-inferred) parameter — excluded from the latent space |

### Fixed Parameters

Use the `fixed` distribution type to hold a parameter constant. Fixed parameters
are excluded from the inferred parameter space but still appear in the full
parameter vector passed to the simulator:

```yaml
simulator:
  _target_: falcon.priors.Product
  priors:
    - ['normal', 0.0, 1.0]     # Inferred
    - ['fixed', 3.14]           # Held constant
    - ['uniform', -1.0, 1.0]   # Inferred
```

## Usage

```python
from falcon.priors import Product

prior = Product(
    priors=[
        ('normal', 0.0, 1.0),
        ('uniform', -10.0, 10.0),
    ]
)

# Sample from prior — output shape is (1000, prior.full_param_dim)
samples = prior.simulate_batch(1000)

# Transform to/from latent space
z = prior.inverse(samples, mode="standard_normal")
x = prior.forward(z, mode="standard_normal")
```

!!! note "param_dim vs full_param_dim"
    `prior.param_dim` is the number of *free* (non-fixed) parameters — the
    dimension of the latent space seen by estimators. `prior.full_param_dim`
    is the total output dimension including `fixed` parameters. The simulator
    always receives the full vector.

## YAML Configuration

### With Flow estimator

```yaml
simulator:
  _target_: falcon.priors.Product
  priors:
    - ['uniform', -100.0, 100.0]
    - ['uniform', -100.0, 100.0]
```

### With GaussianFullCov estimator

```yaml
simulator:
  _target_: falcon.priors.Product
  priors:
    - ['normal', 0.0, 1.0]
    - ['normal', 0.0, 1.0]
```

## Class Reference

::: falcon.priors.product.TransformedPrior
    options:
      show_source: true

::: falcon.priors.product.Product
    options:
      show_source: true
      members:
        - __init__
        - forward
        - inverse
        - simulate_batch
