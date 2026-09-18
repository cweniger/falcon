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
falcon launch -o output/exp01                   # Specify output directory
falcon launch buffer.max_epochs=500              # Override config parameters
falcon launch -c config_amortized                # Use alternate config file

# Sampling (after training)
falcon sample prior -o output/exp01             # Sample from prior
falcon sample posterior -o output/exp01         # Sample from posterior
falcon sample proposal -o output/exp01          # Sample from proposal distribution

# Visualize graph structure
falcon graph                                     # Display ASCII graph visualization

# Run examples
cd examples/01_minimal && falcon launch -o output/run_01
```

## Architecture

### Core Components

**Graph System** (`falcon/core/graph.py`, `falcon/core/deployed_graph.py`):
- `Node`: Represents a random variable with `simulator_cls` (forward model) and `estimator_cls` (posterior learner)
- `Graph`: Manages node relationships, performs topological sorting for execution order
- `CompositeNode`: Factory for multi-output simulator nodes with automatic extraction
- `DeployedGraph`: Orchestrates Ray-based distributed execution of the graph

**Distributed Execution** (`falcon/core/deployed_graph.py`, `falcon/core/raystore.py`):
- `SampleActor` (named `z`): serves prior/proposal/posterior samples of a node; for estimator nodes it holds the best network (`ModelSampler`). `num_actors` sets how many.
- `TrainActor` (named `z/train`, estimator nodes only): trains with `RoundTrainer` and publishes each new best state to the node's sample actors, then runs the discard sweep. Both actors are synchronous; control calls (stop, status, logs) run in a separate Ray concurrency group
- `ray.num_train_gpus` / `ray.num_sample_gpus` set each actor's GPUs (`ray.num_gpus` is a deprecated alias, split evenly); other `ray:` options apply to both
- `DatasetManagerActor`: Centralized dataset orchestration with sample lifecycle (ACTIVE → DISFAVOURED → TOMBSTONE → DELETED) and a separate, fixed purpose per sample (TRAINING or VALIDATION, a `buffer.validation_fraction` share spread evenly by insertion id)

**Engines** (`falcon/core/`, framework-free — no torch imports):
- `BaseEstimator` (`base_estimator.py`): Model contract used unchanged by both actors (build, train_step/evaluate/discard_mask, snapshot/restore, export_state/import_state as numpy trees, sample_prior/sample with an explicit rng); `RoundConfig` holds the loop parameters
- `RoundTrainer` (`round_trainer.py`): Round-based training loop (fixed data per round, best network validated at round start, epochs until convergence, acceptance test, publish, discard sweep after accepted rounds; see `docs/training.md`); owns round counters, best state and the checkpoint
- `ModelSampler` (`model_sampler.py`): Installs published states and falls back to the prior (no best state yet / `prior_rounds`)
- `state_io.py`: State trees (nested dicts of numpy arrays and plain values) and `graph/<node>/best_state.npz`

**Estimators** (`falcon/estimators/`):
- `TorchModel` (`torch_model.py`): Base class for torch estimators; network groups, state conversion, seeded sampling, legacy `.pth` loading
- `Flow` (`flow.py`): Flow-based posterior estimation using conditional + marginal flow pair with importance sampling
- `FlowDensity` (`flow_density.py`): Flow network wrapper around `sbi.neural_nets` (the only file importing `sbi`)
- `GaussianFullCov` (`gaussian_fullcov.py`): Full covariance Gaussian posterior (requires a `TransformedPrior` such as `Product`)
- `EmbeddedPosterior` (`embedded_posterior.py`): Wrapper combining embedding network with posterior model
- `networks.py`: MLP builder utility

**Priors** (`falcon/priors/`):
- `Product` (`product.py`): Product of independent marginals; supports `mode="hypercube"` (for Flow) and `mode="standard_normal"` (for Gaussian), plus `"fixed"` parameters

**Embeddings** (`falcon/embeddings/`):
- `instantiate_embedding` (`builder.py`): Declarative embedding builder supporting nested configurations
- `LazyOnlineNorm`, `DiagonalWhitener` (`norms.py`): Online normalization utilities
- `DynamicSVD` (`svd.py`): Streaming SVD with Procrustes-stabilized output and optional whitening
- `LazyBuffersMixin` (`lazy.py`): Buffers created from the first data; fresh modules can load trained state

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
- `buffer`: Sample management (min/max samples, simulate interval, dump settings)
- `graph`: Node definitions with simulators, estimators, and dependencies
- `sample`: Sampling parameters (prior, posterior, proposal)
- `ray`: Ray initialization (CPU/GPU allocation per node)

### Graph Definition Pattern

```yaml
graph:
  theta:                          # Latent parameters node
    evidence: [x]                 # Inferred from observation x
    simulator:                    # Prior distribution
      _target_: falcon.priors.Product
      priors:
        - ['uniform', -100.0, 100.0]
    estimator:                    # Posterior network
      _target_: falcon.estimators.Flow
      loop:                       # Training loop config
        max_epochs: 300
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
      num_train_gpus: 0           # train actor (z/train)
      num_sample_gpus: 0          # sample actor (z)

  x:                              # Observation node
    parents: [theta]              # Depends on theta
    simulator:                    # Forward model (user-defined)
      _target_: model.Simulator
    observed: "./data/obs.npz['x']"  # NPZ key extraction syntax
```

### Key Design Patterns

- **Lazy Loading**: Classes defined as strings (`_target_`), instantiated at runtime via `LazyLoader` (`falcon/core/utils.py`)
- **Ray Actors**: All distributed computation uses Ray actor model; training and sampling of a node run in separate actors
- **Framework-free core**: only numpy crosses actor boundaries; torch lives in the estimators (JAX estimators are planned)
- **Declarative Configuration**: YAML drives model/training decisions
- **Optional Dependencies**: `wandb` uses try/except in `wandb_logger.py`; `sbi` is isolated to `flow_density.py`

## Output Structure

```
{run_dir}/
├── graph/                      # Trained models and logs
│   ├── driver/                 # Driver log and metrics
│   ├── {node_name}/            # Per-node directories (sample actor stream)
│   │   ├── best_state.npz      # Best network (state tree), round counters
│   │   ├── training_history.npz
│   │   ├── output.log          # Sample actor log
│   │   ├── metrics/            # Metric history (chunk_*.npz)
│   │   └── train/              # Train actor stream: output.log, metrics/
├── samples/                    # Generated samples
│   └── posterior/              # Posterior sample files
│       ├── 000000.npz
│       └── 000001.npz
├── buffer/
│   └── snapshots/              # Buffer snapshots (when snapshot_every > 0)
│       └── 000000.npz
└── config.yml                 # Saved configuration
```

## Key Files

- `falcon/cli.py`: Entry point, implements `launch_mode`, `sample_mode`, `graph_mode`
- `falcon/core/graph.py`: Graph, Node, and CompositeNode definitions
- `falcon/core/deployed_graph.py`: Runtime execution with Ray (sample/train actors)
- `falcon/core/base_estimator.py`: Model contract of all estimators
- `falcon/core/round_trainer.py`: Round-based training loop
- `falcon/core/logger.py`: Unified logging system with pluggable backends
- `falcon/core/run_loader.py`: Unified `Run` loader for post-training analysis
- `falcon/estimators/torch_model.py`: Base class for torch estimators
- `falcon/estimators/flow.py`: Flow-based posterior estimation (conditional + marginal flows)
- `falcon/estimators/flow_density.py`: sbi-backed flow networks (only sbi import point)
- `falcon/estimators/gaussian_fullcov.py`: Full covariance Gaussian posterior estimation
- `falcon/priors/product.py`: Product prior with latent space transformations (hypercube and standard_normal modes)
- `falcon/embeddings/builder.py`: Declarative embedding pipeline builder
- `falcon/interactive.py`: Interactive TUI display for launch mode
- `examples/`: 01_minimal, 02_bimodal, 03_composite, 04_gaussian, 05_linear_regression
