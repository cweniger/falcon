# Falcon - Distributed Dynamic Simulation-Based Inference

*Falcon* is a Python framework for **simulation-based inference (SBI)** that enables adaptive learning of complex conditional distributions. Built on top of PyTorch, Ray, and sbi, *Falcon* provides a declarative approach to building probabilistic models with automatic parallelization and experiment tracking.

## Key Features

- **Declarative Model Definition**: Define complex probabilistic models using YAML configuration
- **Graph-Based Architecture**: Express dependencies between random variables as a directed graph
- **Adaptive Sampling**: Automatically manage simulation buffers with configurable resampling
- **Distributed Computing**: Built-in parallelization using Ray actors
- **Pluggable Estimators**: Modular design for different inference algorithms
- **Experiment Tracking**: Integrated WandB and local file logging

## Installation

```bash
git clone https://github.com/cweniger/falcon.git
cd falcon
pip install .
```

**Dependencies** (automatically installed):

| Package | Purpose |
|---------|---------|
| `torch>=2.0.0` | Deep learning framework |
| `numpy` | Numerical computing |
| `ray` | Distributed computing |
| `sbi` | Simulation-based inference |
| `omegaconf` | Configuration management |
| `wandb>=0.15.0` | Experiment tracking |
| `coolname` | Auto-naming for runs |

**Optional**: `pip install "falcon[monitor]"` for TUI dashboard

## Quick Start

```bash
cd examples/01_minimal
falcon launch --run-dir outputs/run_01
falcon sample posterior --run-dir outputs/run_01
```

## Command-Line Interface

| Command | Description |
|---------|-------------|
| `falcon launch` | Run training |
| `falcon sample prior\|posterior\|proposal` | Generate samples |
| `falcon graph` | Visualize graph structure |
| `falcon monitor` | Real-time TUI dashboard |

```bash
falcon launch --run-dir DIR              # Specify output directory
falcon launch --config-name NAME         # Use alternate config file
falcon launch key=value                  # Override config parameters
falcon sample posterior --run-dir DIR    # Sample from trained model
```

## Core Concepts

### Computational Graph

Falcon models are defined as directed acyclic graphs where:
- **Nodes** represent random variables
- **Edges** define dependencies between variables
- **Simulators** define forward models (priors or conditional distributions)
- **Estimators** learn inverse mappings (posteriors)

```yaml
graph:
  z:                              # Latent parameters (to be inferred)
    evidence: [x]                 # Inferred from observation x
    simulator: ...                # Prior p(z)
    estimator: ...                # Learns p(z|x)

  x:                              # Observation
    parents: [z]                  # Depends on z
    simulator: ...                # Forward model p(x|z)
    observed: "./data/obs.npy"    # Observed data
```

### Configuration Structure

```yaml
logging:                          # Experiment tracking
  wandb:
    enabled: false
    project: my_project
  local:
    enabled: true
    dir: ${paths.graph}

paths:                            # Directory layout
  import: "./src"                 # User code location
  buffer: ${run_dir}/sim_dir      # Simulation data
  graph: ${run_dir}/graph_dir     # Trained models
  samples: ${run_dir}/samples_dir # Generated samples

buffer:                           # Sample management
  min_training_samples: 4096
  max_training_samples: 32768
  resample_batch_size: 128
  resample_interval: 10

graph:                            # Model definition
  # Node definitions...

sample:                           # Sampling settings
  posterior:
    n: 1000
```

### Simulators

Simulators define probability distributions. They must implement `simulate_batch`:

```python
class MySimulator:
    def simulate_batch(self, batch_size: int, **parent_values) -> torch.Tensor:
        """Generate samples conditioned on parent values."""
        z = parent_values['z']  # Parent node values
        return forward_model(z)
```

**Built-in simulators:**
- `falcon.contrib.HypercubeMappingPrior` - Configurable prior distributions

### Estimators

Estimators learn conditional distributions from simulated data. They must implement the `BaseEstimator` interface.

**Built-in estimators:**
- `falcon.contrib.SNPE_A` - Sequential Neural Posterior Estimation ([detailed docs](docs/estimators/SNPE_A.md))

### Buffer Management

The buffer controls how training data is collected and managed:

| Parameter | Description |
|-----------|-------------|
| `min_training_samples` | Minimum samples before training starts |
| `max_training_samples` | Maximum buffer size |
| `resample_batch_size` | New samples per resampling step |
| `resample_interval` | Epochs between resampling |
| `keep_resampling` | Continue after max reached |
| `validation_window_size` | Validation split size |

### Prior Distributions

`HypercubeMappingPrior` supports these distribution types:

```yaml
priors:
  - ['uniform', low, high]
  - ['normal', mean, std]
  - ['triangular', low, mode, high]
  - ['cosine', low, high]
  - ['sine', low, high]
  - ['uvol', low, high]              # Uniform-in-volume
```

## Creating a Model

### 1. Define Your Simulator

```python
# src/model.py
import torch

class Simulate:
    def __init__(self, noise_scale: float = 0.1):
        self.noise_scale = noise_scale

    def simulate_batch(self, batch_size: int, z: torch.Tensor) -> torch.Tensor:
        return z + torch.randn_like(z) * self.noise_scale

class E(torch.nn.Module):
    """Embedding network."""
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.LazyLinear(64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
```

### 2. Create Configuration

```yaml
# config.yaml
logging:
  wandb:
    enabled: false
  local:
    enabled: true
    dir: ${paths.graph}

paths:
  import: "./src"
  buffer: ${run_dir}/sim_dir
  graph: ${run_dir}/graph_dir
  samples: ${run_dir}/samples_dir

buffer:
  min_training_samples: 2048
  max_training_samples: 8192
  resample_batch_size: 128
  resample_interval: 10

graph:
  z:
    evidence: [x]
    simulator:
      _target_: falcon.contrib.HypercubeMappingPrior
      priors:
        - ['uniform', -5.0, 5.0]
        - ['uniform', -5.0, 5.0]
    estimator:
      _target_: falcon.contrib.SNPE_A
      # See docs/estimators/SNPE_A.md for configuration options

  x:
    parents: [z]
    simulator:
      _target_: model.Simulate
      noise_scale: 0.1
    observed: "./data/observations.npy"

sample:
  posterior:
    n: 1000
```

### 3. Prepare Data and Run

```python
import numpy as np
np.save("data/observations.npy", np.array([1.5, -0.3]))
```

```bash
falcon launch --run-dir outputs/my_run
falcon sample posterior --run-dir outputs/my_run
```

## Advanced Features

### Multi-Node Graphs

```yaml
graph:
  z:
    evidence: [x]
    simulator: ...
    estimator: ...

  signal:
    parents: [z]
    simulator:
      _target_: model.Signal

  noise:
    simulator:
      _target_: model.Noise

  x:
    parents: [signal, noise]
    simulator:
      _target_: model.Combine
    observed: "./data/obs.npy"
```

### GPU Allocation

```yaml
graph:
  z:
    ray:
      num_gpus: 1      # Full GPU
      # num_gpus: 0.5  # Fractional GPU (multiple nodes per GPU)
```

### Intermediate Sample Dumping

```yaml
buffer:
  dump:
    enabled: true
    path: sample_{step}.joblib
```

## Output Structure

```
{run_dir}/
├── sim_dir/                    # Simulation buffer
│   └── samples_*.pt
├── graph_dir/                  # Trained models
│   ├── graph.pkl
│   ├── {node}/estimator.pt
│   ├── output.log
│   └── metrics/
├── samples_dir/                # Generated samples
│   └── posterior/{timestamp}/
└── config.yaml
```

## Examples

| Example | Description |
|---------|-------------|
| `01_minimal` | Basic 3-parameter inference |
| `02_bimodal` | 10D bimodal posterior with training strategies |
| `03_composite` | Multi-node graph with image embeddings |

## Documentation

- [SNPE_A Estimator](docs/estimators/SNPE_A.md) - Detailed configuration reference

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Out of Memory | Reduce `max_training_samples` or batch size |
| Slow Training | Enable GPU: `ray.num_gpus: 1` |
| Import Errors | Check `paths.import` points to your code |
| Monitor not working | `pip install "falcon[monitor]"` |

## Citation

```bibtex
@software{falcon2024,
  title = {Falcon: Distributed Dynamic Simulation-Based Inference},
  author = {Weniger, Christoph},
  year = {2024},
  url = {https://github.com/cweniger/falcon}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.
