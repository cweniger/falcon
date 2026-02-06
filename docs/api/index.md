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
│   ├── gaussian        # Gaussian posterior estimation
│   └── flow_density    # Normalizing flow networks
├── priors/            # Prior distributions
│   ├── hypercube       # Hypercube mapping prior
│   └── product         # Product of marginals
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
| [`Gaussian`](gaussian.md) | Full covariance Gaussian posterior |
| [`FlowDensity`](flow-density.md) | Normalizing flow `nn.Module` (internal) |

## Priors

| Class | Description |
|-------|-------------|
| [`Hypercube`](hypercube.md) | Hypercube-to-target distribution mapping |
| [`Product`](product.md) | Product of independent marginals with latent space transforms |

## Embeddings

| Class / Function | Description |
|------------------|-------------|
| [`instantiate_embedding`](embeddings.md) | Declarative embedding pipeline builder |
| [`LazyOnlineNorm`](embeddings.md#falcon.embeddings.norms.LazyOnlineNorm) | Online normalization |
| [`DiagonalWhitener`](embeddings.md#falcon.embeddings.norms.DiagonalWhitener) | Diagonal whitening |
| [`PCAProjector`](embeddings.md#falcon.embeddings.svd.PCAProjector) | Streaming PCA projector |

## Quick Import

```python
import falcon

# Core
from falcon import Graph, Node, CompositeNode, DeployedGraph

# Estimators
from falcon.estimators import Flow, Gaussian

# Priors
from falcon.priors import Hypercube, Product

# Embeddings
from falcon.embeddings import instantiate_embedding, LazyOnlineNorm, PCAProjector

# Utilities
from falcon import read_run, load_run, read_samples
```
