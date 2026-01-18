# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Falcon is a Python framework for simulation-based inference (SBI) that enables adaptive learning of complex conditional distributions. Built on PyTorch, Ray, and the sbi library, it provides a declarative YAML-based approach to defining probabilistic models with automatic parallelization and WandB experiment tracking.

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

# Real-time monitoring (requires: pip install "falcon[monitor]")
falcon monitor                                   # TUI dashboard for training progress

# Run examples
cd examples/01_minimal && falcon launch --run-dir outputs/run_01
```

## Architecture

### Core Components

**Graph System** (`falcon/core/graph.py`, `falcon/core/deployed_graph.py`):
- `Node`: Represents a random variable with `simulator_cls` (forward model) and `estimator_cls` (posterior learner)
- `Graph`: Manages node relationships, performs topological sorting for execution order
- `DeployedGraph`: Orchestrates Ray-based distributed execution of the graph

**Distributed Execution** (`falcon/core/raystore.py`):
- `NodeWrapper`: Ray actor wrapping individual nodes for async training
- `DatasetManagerActor`: Centralized dataset orchestration with sample lifecycle (VALIDATION → TRAINING → DISFAVOURED → TOMBSTONE)

**Inference** (`falcon/contrib/SNPE_A.py`):
- Sequential Neural Posterior Estimation with support for multiple flow architectures (NSF, MAF, NAF, Zuko flows)
- Features: amortization (gamma parameter), parameter normalization, early stopping, LR scheduling

**Embeddings** (`falcon/contrib/torch_embedding.py`):
- Declarative embedding builder supporting nested configurations
- Compatible with external libraries (timm, transformers)

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
      _target_: falcon.contrib.HypercubeMappingPrior
      priors:
        - ['uniform', -100.0, 100.0]
    estimator:                    # Posterior network
      _target_: falcon.contrib.SNPE_A
      loop:                       # Training loop config
        num_epochs: 300
        batch_size: 128
      network:                    # Network config
        net_type: nsf
        embedding:
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

- **Lazy Loading**: Classes defined as strings (`_target_`), instantiated at runtime via `LazyLoader`
- **Ray Actors**: All distributed computation uses Ray actor model
- **Declarative Configuration**: YAML drives model/training decisions
- **Async Operations**: asyncio for efficient resource utilization in actors

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
└── config.yaml                 # Saved configuration
```

## Key Files

- `falcon/cli.py`: Entry point, implements `launch_mode` and `sample_mode`
- `falcon/core/graph.py`: Graph and Node class definitions
- `falcon/core/deployed_graph.py`: Runtime execution with Ray
- `falcon/contrib/SNPE_A.py`: Main inference algorithm
- `falcon/contrib/hypercubemappingprior.py`: Prior distribution transformations
- `examples/01_minimal/`: Best starting point for understanding the framework
