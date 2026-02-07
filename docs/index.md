# Falcon

**A Python framework for simulation-based inference with adaptive learning.**

Falcon enables adaptive learning of complex conditional distributions through a declarative YAML-based approach. Built on PyTorch, Ray, and the sbi library, it provides automatic parallelization and experiment tracking.

## Key Features

- **Declarative Configuration**: Define probabilistic models using intuitive YAML syntax
- **Automatic Parallelization**: Distributed execution via Ray actors
- **Adaptive Learning**: Sequential neural posterior estimation with importance sampling
- **Experiment Tracking**: Built-in WandB integration for monitoring training

## Quick Example

```yaml
graph:
  theta:
    evidence: [x]
    simulator:
      _target_: falcon.priors.Hypercube
      priors:
        - ['uniform', -10.0, 10.0]
    estimator:
      _target_: falcon.estimators.Flow

  x:
    parents: [theta]
    simulator:
      _target_: model.Simulator
    observed: "./data/obs.npz['x']"
```

```bash
falcon launch --run-dir outputs/run_01
```

## Installation

```bash
pip install falcon-sbi
```

Or install from source:

```bash
git clone https://github.com/cweniger/falcon.git
cd falcon
pip install -e .
```

## Next Steps

- [Getting Started](getting-started.md): Walk through your first Falcon project
- [Configuration](configuration.md): Learn the YAML configuration format
- [API Reference](api/index.md): Explore the Python API
