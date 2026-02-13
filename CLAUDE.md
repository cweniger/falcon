# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Falcon is a Python framework for simulation-based inference (SBI) that enables adaptive learning of complex conditional distributions. Built on PyTorch, Ray, and the sbi library, it provides a declarative YAML-based approach to defining probabilistic models with automatic parallelization and optional WandB experiment tracking.

## Common Commands

```bash
# Install from source
pip install .

# Training
falcon launch                                    # Run with default config
falcon launch --run-dir outputs/exp01            # Specify output directory
falcon launch buffer.num_epochs=500              # Override config parameters
falcon launch --config-name config_amortized     # Use alternate config file

# Sampling (after training)
falcon sample prior --run-dir outputs/exp01      # Sample from prior
falcon sample posterior --run-dir outputs/exp01  # Sample from posterior
falcon sample proposal --run-dir outputs/exp01   # Sample from proposal distribution

# Visualize graph structure
falcon graph                                     # Display ASCII graph visualization

# Real-time monitoring (requires: pip install "falcon-sbi[monitor]")
falcon monitor                                   # TUI dashboard for training progress

# Run examples
cd examples/01_minimal && falcon launch --run-dir outputs/run_01
```

## Architecture

### Core Components

**Graph System** (`falcon/core/graph.py`, `falcon/core/deployed_graph.py`):
- `Node`: Represents a random variable with `simulator_cls` (forward model) and `estimator_cls` (posterior learner)
- `Graph`: Manages node relationships, performs topological sorting for execution order
- `CompositeNode`: Factory for multi-output simulator nodes with automatic extraction
- `DeployedGraph`: Orchestrates Ray-based distributed execution of the graph

**Distributed Execution** (`falcon/core/raystore.py`):
- `NodeWrapper`: Ray actor wrapping individual nodes for async training
- `DatasetManagerActor`: Centralized dataset orchestration with sample lifecycle (VALIDATION → TRAINING → DISFAVOURED → TOMBSTONE)

**Estimators** (`falcon/estimators/`):
- `BaseEstimator` (`falcon/core/base_estimator.py`): Abstract interface defining train/sample/save/load contract
- `StepwiseEstimator`, `LossBasedEstimator` (`base.py`): Base classes for epoch-based training with early stopping
- `Flow` (`flow.py`): Flow-based posterior estimation using conditional + marginal flow pair with importance sampling
- `FlowDensity` (`flow_density.py`): Flow network wrapper around `sbi.neural_nets` (the only file importing `sbi`)
- `Gaussian` (`gaussian.py`): Factory creating a `LossBasedEstimator` with full covariance Gaussian posterior
- `EmbeddedPosterior` (`embedded_posterior.py`): Wrapper combining embedding network with posterior model
- `networks.py`: MLP builder utility

**Priors** (`falcon/priors/`):
- `Hypercube` (`hypercube.py`): Bidirectional hypercube-to-target distribution mapping
- `Product` (`product.py`): Product of independent marginals with fixed parameter support

**Embeddings** (`falcon/embeddings/`):
- `instantiate_embedding` (`builder.py`): Declarative embedding builder supporting nested configurations
- `LazyOnlineNorm`, `DiagonalWhitener` (`norms.py`): Online normalization utilities
- `PCAProjector` (`svd.py`): Streaming PCA for dimensionality reduction

**Logging** (`falcon/core/logger.py`, `falcon/core/local_logger.py`, `falcon/core/wandb_logger.py`):
- `Logger`: Unified logging with pluggable backends
- `LocalFileBackend`: Chunked NPZ metric storage
- `WandBBackend`: Optional WandB integration (graceful fallback if wandb not installed)

**Run Analysis** (`falcon/core/run_reader.py`, `falcon/core/run_loader.py`, `falcon/core/samples_reader.py`):
- `read_run(path)`: Lazy-loaded metric reader for locally logged training runs
- `load_run(path)`: Unified `Run` object with config, metrics, samples, and observations
- `read_samples(path)`: Sample set reader with indexing, key access, and filtering

### Configuration System

Uses OmegaConf for configuration management. Key YAML sections:
- `logging`: WandB (`logging.wandb`) and local file (`logging.local`) logging
- `paths`: import path, buffer directory, graph directory, samples directory
- `buffer`: Sample management (min/max samples, resample interval, dump settings)
- `graph`: Node definitions with simulators, estimators, and dependencies
- `sample`: Sampling parameters (prior, posterior, proposal)
- `ray`: Ray initialization (CPU/GPU allocation per node)

### Graph Definition Pattern

```yaml
graph:
  theta:                          # Latent parameters node
    evidence: [x]                 # Inferred from observation x
    simulator:                    # Prior distribution
      _target_: falcon.priors.Hypercube
      priors:
        - ['uniform', -100.0, 100.0]
    estimator:                    # Posterior network
      _target_: falcon.estimators.Flow
      loop:                       # Training loop config
        num_epochs: 300
        batch_size: 128
      network:                    # Network config
        net_type: nsf
      embedding:                  # Embedding config (sibling of network)
        _target_: model.E
        _input_: [x]
      optimizer:                  # Optimizer config
        lr: 0.01
      inference:                  # Inference config
        gamma: 0.5
    ray:
      num_gpus: 0

  x:                              # Observation node
    parents: [theta]              # Depends on theta
    simulator:                    # Forward model (user-defined)
      _target_: model.Simulator
    observed: "./data/obs.npz['x']"  # NPZ key extraction syntax
```

### Key Design Patterns

- **Lazy Loading**: Classes defined as strings (`_target_`), instantiated at runtime via `LazyLoader` (`falcon/core/utils.py`)
- **Ray Actors**: All distributed computation uses Ray actor model
- **Declarative Configuration**: YAML drives model/training decisions
- **Async Operations**: asyncio for efficient resource utilization in actors
- **Optional Dependencies**: `wandb` uses try/except in `wandb_logger.py`; `sbi` is isolated to `flow_density.py`

## Output Structure

```
{run_dir}/
├── sim_dir/                    # Simulation buffer (samples_*.pt)
├── graph_dir/                  # Trained models and logs
│   ├── graph.pkl               # Serialized graph structure
│   ├── {node_name}/            # Per-node directories
│   │   └── estimator.pt        # Network weights
│   ├── output.log              # Training logs
│   └── metrics/                # Metric history (chunk_*.npz)
├── samples_dir/                # Generated samples
│   └── posterior/{timestamp}/  # Timestamped sample batches
│       └── 000000.npz
└── config.yml                 # Saved configuration
```

## Key Files

- `falcon/cli.py`: Entry point, implements `launch_mode`, `sample_mode`, `graph_mode`, `monitor_mode`
- `falcon/core/graph.py`: Graph, Node, and CompositeNode definitions
- `falcon/core/deployed_graph.py`: Runtime execution with Ray
- `falcon/core/base_estimator.py`: Abstract estimator interface
- `falcon/core/logger.py`: Unified logging system with pluggable backends
- `falcon/core/run_loader.py`: Unified `Run` loader for post-training analysis
- `falcon/estimators/flow.py`: Flow-based posterior estimation (conditional + marginal flows)
- `falcon/estimators/flow_density.py`: sbi-backed flow networks (only sbi import point)
- `falcon/estimators/gaussian.py`: Gaussian posterior estimation via LossBasedEstimator
- `falcon/priors/hypercube.py`: Hypercube mapping prior distribution
- `falcon/priors/product.py`: Product prior with latent space transformations
- `falcon/embeddings/builder.py`: Declarative embedding pipeline builder
- `falcon/interactive.py`: Interactive TUI display for launch mode
- `examples/`: 01_minimal, 02_bimodal, 03_composite, 04_gaussian, 05_linear_regression
