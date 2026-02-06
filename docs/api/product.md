# Product Prior

Product of independent marginal distributions with latent space transformations.

## Overview

`Product` defines a prior as a product of independent 1D marginals, each with a
bijective map to a chosen latent space. It extends the abstract base class
`TransformedPrior`, which defines the `forward`/`inverse` interface.

`Product` supports two latent-space modes:

- **`"hypercube"`**: Maps to/from a bounded hypercube (used with [Flow](flow.md))
- **`"standard_normal"`**: Maps to/from standard normal space (used with [Gaussian](gaussian.md))

## Supported Distributions

| Type | Parameters | Description |
|------|------------|-------------|
| `uniform` | `low`, `high` | Uniform distribution |
| `normal` | `mean`, `std` | Gaussian distribution |
| `cosine` | `low`, `high` | Cosine-weighted distribution |
| `sine` | `low`, `high` | Sine-weighted distribution |
| `uvol` | `low`, `high` | Uniform-in-volume |
| `triangular` | `a`, `c`, `b` | Triangular distribution (min, mode, max) |
| `lognormal` | `mean`, `std` | Log-normal distribution |
| `fixed` | `value` | Fixed (non-inferred) parameter |

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

# Sample from prior
samples = prior.simulate_batch(1000)

# Transform to/from latent space
z = prior.inverse(samples, mode="standard_normal")
x = prior.forward(z, mode="standard_normal")
```

## YAML Configuration

### With Flow estimator

```yaml
simulator:
  _target_: falcon.priors.Product
  priors:
    - ['uniform', -100.0, 100.0]
    - ['uniform', -100.0, 100.0]
```

### With Gaussian estimator

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
