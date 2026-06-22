# API Reference

This section documents Falcon's Python API.

## Package Structure

```
falcon/
├── core/              # Core framework
│   ├── graph           # Graph and Node definitions
│   ├── deployed_graph  # Runtime execution
│   └── base_estimator  # Estimator interface
├── estimators/        # Posterior estimation
│   ├── flow            # Flow-based posterior estimation
│   ├── gaussian_fullcov  # Gaussian posterior estimation
│   ├── flow_density    # Normalizing flow networks
│   ├── stepwise_base   # Base training loop classes
│   ├── networks        # MLP builder utilities
│   └── embedded_posterior  # Embedding + posterior wrapper
├── priors/            # Prior distributions
│   └── product         # Product of independent marginals
└── embeddings/        # Observation embeddings
    ├── builder         # Declarative embedding pipelines
    ├── norms           # Online normalization utilities
    └── svd             # Streaming PCA
```

## Core Classes

| Class | Description |
|-------|-------------|
| [`Graph`](graph.md) | Container for computational graph nodes |
| [`Node`](graph.md#falcon.core.graph.Node) | Single random variable in the graph |
| [`DeployedGraph`](deployed-graph.md) | Runtime orchestration with Ray |
| [`BaseEstimator`](base-estimator.md) | Abstract interface for estimators |

## Estimators

| Class | Description |
|-------|-------------|
| [`Flow`](flow.md) | Flow-based posterior estimation (normalizing flows) |
| [`GaussianFullCov`](gaussian.md) | Full covariance Gaussian posterior |
| [`FlowDensity`](flow-density.md) | Normalizing flow `nn.Module` (internal) |

## Priors

| Class | Description |
|-------|-------------|
| [`Product`](product.md) | Product of independent marginals with latent space transforms |

## Embeddings

| Class / Function | Description |
|------------------|-------------|
| [`instantiate_embedding`](embeddings.md) | Declarative embedding pipeline builder |
| [`LazyOnlineNorm`](embeddings.md#falcon.embeddings.norms.LazyOnlineNorm) | Online normalization |
| [`DiagonalWhitener`](embeddings.md#falcon.embeddings.norms.DiagonalWhitener) | Diagonal whitening |
| [`DynamicSVD`](embeddings.md#falcon.embeddings.svd.DynamicSVD) | Streaming SVD with Procrustes stabilization |

## Quick Import

```python
import falcon

# Core
from falcon import Graph, Node, CompositeNode, DeployedGraph

# Estimators
from falcon.estimators import Flow, GaussianFullCov

# Priors
from falcon.priors import Product

# Embeddings
from falcon.embeddings import instantiate_embedding, LazyOnlineNorm, DynamicSVD

# Utilities
from falcon import read_run, load_run, read_samples
```
