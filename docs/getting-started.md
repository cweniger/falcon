# Getting Started

This guide walks you through setting up and running your first Falcon project.

## Prerequisites

- Python 3.9+
- PyTorch 2.0+

## Installation

```bash
pip install falcon-sbi
```

## Project Structure

A typical Falcon project has this structure:

```
my_project/
├── config.yml      # Graph and training configuration
├── model.py         # Your simulator and embedding definitions
└── data/
    └── obs.npz      # Observed data (optional)
```

## Minimal Example

### 1. Define Your Simulator

Create `model.py` with a forward model:

```python
import numpy as np

class Simulator:
    """Simple Gaussian simulator."""

    def simulate_batch(self, batch_size, theta):
        noise = np.random.randn(*theta.shape) * 0.1
        return theta + noise
```

### 2. Create Configuration

Create `config.yml`:

```yaml
logging:
  wandb:
    enabled: false
  local:
    enabled: true

paths:
  import: "."

buffer:
  min_samples: 1000
  max_samples: 10000
  validation_samples: 256
  simulate_count: 64
  simulate_interval: 1

graph:
  theta:
    evidence: [x]
    simulator:
      _target_: falcon.priors.Hypercube
      priors:
        - ['uniform', -10.0, 10.0]
    estimator:
      _target_: falcon.estimators.Flow
      loop:
        max_epochs: 100
      network:
        net_type: zuko_nice

  x:
    parents: [theta]
    simulator:
      _target_: model.Simulator
    observed: "./data/obs.npz['x']"
```

### 3. Prepare Observation Data

```python
import numpy as np

# Generate synthetic observation
true_theta = 2.5
obs = true_theta + np.random.randn(100) * 0.1

np.savez("data/obs.npz", x=obs)
```

### 4. Run Training

```bash
falcon launch -o outputs/run_01
```

### 5. Sample from Posterior

```bash
falcon sample posterior -o outputs/run_01
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `falcon launch` | Start training |
| `falcon sample prior` | Sample from prior |
| `falcon sample posterior` | Sample from learned posterior |
| `falcon sample proposal` | Sample from proposal distribution |
| `falcon graph` | Display graph structure |

## Next Steps

- Learn about [Configuration](configuration.md) options
- Explore the [API Reference](api/index.md)
