# API Reference

This section documents Falcon's Python API.

## Package Structure

```
falcon/
├── core/           # Core framework
│   ├── graph       # Graph and Node definitions
│   ├── deployed_graph  # Runtime execution
│   └── base_estimator  # Estimator interface
└── contrib/        # Built-in implementations
    ├── SNPE_A      # Neural posterior estimation
    ├── hypercubemappingprior  # Prior distributions
    └── flow        # Normalizing flows
```

## Core Classes

| Class | Description |
|-------|-------------|
| [`Graph`](graph.md) | Container for computational graph nodes |
| [`Node`](graph.md#falcon.core.graph.Node) | Single random variable in the graph |
| [`DeployedGraph`](deployed-graph.md) | Runtime orchestration with Ray |
| [`BaseEstimator`](base-estimator.md) | Abstract interface for estimators |

## Contrib Classes

| Class | Description |
|-------|-------------|
| [`SNPE_A`](snpe-a.md) | Sequential Neural Posterior Estimation |
| [`HypercubeMappingPrior`](hypercube-prior.md) | Flexible prior distributions |
| [`Flow`](flow.md) | Normalizing flow networks |

## Quick Import

```python
import falcon

# Core
from falcon import Graph, Node, CompositeNode, DeployedGraph

# Contrib
from falcon.contrib import SNPE_A, HypercubeMappingPrior, Flow

# Utilities
from falcon import read_run, load_run, read_samples
```
