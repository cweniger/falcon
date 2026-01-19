# HypercubeMappingPrior

Flexible prior distributions with hypercube mapping.

## Overview

`HypercubeMappingPrior` maps between a hypercube domain and various target
distributions. This enables uniform treatment of different prior types during
training while preserving the original distribution semantics.

## Supported Distributions

| Type | Parameters | Description |
|------|------------|-------------|
| `uniform` | `low`, `high` | Uniform distribution |
| `normal` | `mean`, `std` | Gaussian distribution |
| `cosine` | `low`, `high` | Distribution with pdf proportional to sin(angle) |
| `sine` | `low`, `high` | Inverse sine mapping |
| `uvol` | `low`, `high` | Uniform-in-volume |
| `triangular` | `a`, `c`, `b` | Triangular distribution (min, mode, max) |

## Usage

```python
from falcon.contrib import HypercubeMappingPrior

# Define priors for 3 parameters
prior = HypercubeMappingPrior(
    priors=[
        ('uniform', -10.0, 10.0),
        ('normal', 0.0, 1.0),
        ('triangular', -1.0, 0.0, 1.0),
    ]
)

# Sample from prior
samples = prior.simulate_batch(1000)

# Transform to/from hypercube space
u = prior.inverse(samples)   # To hypercube
x = prior.forward(u)         # From hypercube
```

## YAML Configuration

```yaml
simulator:
  _target_: falcon.contrib.HypercubeMappingPrior
  priors:
    - ['uniform', -10.0, 10.0]
    - ['normal', 0.0, 1.0]
  hypercube_range: [-2, 2]
```

## Class Reference

::: falcon.contrib.hypercubemappingprior.HypercubeMappingPrior
    options:
      show_source: true
      members:
        - __init__
        - forward
        - inverse
        - simulate_batch
