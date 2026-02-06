# Falcon

[![Documentation](https://img.shields.io/badge/docs-cweniger.github.io%2Ffalcon-blue)](https://cweniger.github.io/falcon/)

Falcon is a CLI-driven Python framework for **simulation-based inference (SBI)** with large, expensive simulators. Born in astrophysics, built for any domain with complex forward models — break your model into components and Falcon jointly infers their parameters.

- **Composable** — define multi-component models as a graph of simulators in YAML, each wrapped with a thin Python interface, regardless of framework.
- **Adaptive** — steers simulations toward high-posterior regions as training progresses, focusing compute where it matters.
- **Concurrent** — trains neural posterior estimators across heterogeneous parameter blocks in parallel, using Ray for distributed execution.
- **Batteries included** — ships with neural spline flows, data embeddings (including CNN/transformer support), and built-in experiment tracking via WandB.

## Installation

```bash
git clone https://github.com/cweniger/falcon.git
cd falcon
pip install .
```

## Quick Start

Run the minimal example (a 3-parameter Gaussian inference problem):

```bash
cd examples/01_minimal
falcon launch --run-dir outputs/run_01
falcon sample posterior --run-dir outputs/run_01
```

This trains a neural posterior estimator on simulated data, then draws 1000 posterior samples. Results are saved under `outputs/run_01/`.

## How It Works

You define a directed graph of random variables in `config.yaml`. Each node has a **simulator** (forward model) and optionally an **estimator** (learned posterior). Falcon iterates between simulating data and training the estimator, automatically managing the sample buffer.

```yaml
graph:
  z:                                    # Latent parameters
    evidence: [x]
    simulator:
      _target_: falcon.contrib.HypercubeMappingPrior
      priors:
        - ['uniform', -5.0, 5.0]
    estimator:
      _target_: falcon.contrib.SNPE_A

  x:                                    # Observations
    parents: [z]
    simulator:
      _target_: model.Simulate
    observed: "./data/obs.npz['x']"
```

## CLI

```bash
falcon launch [--run-dir DIR] [--config-name NAME] [key=value ...]
falcon sample prior|posterior|proposal --run-dir DIR
falcon graph                            # Visualize graph structure
falcon monitor                          # Real-time TUI dashboard (requires pip install "falcon[monitor]")
```

## Examples

| Example | Description |
|---------|-------------|
| [`01_minimal`](examples/01_minimal) | Basic 3-parameter inference |
| [`02_bimodal`](examples/02_bimodal) | 10D bimodal posterior with training strategies |
| [`03_composite`](examples/03_composite) | Multi-node graph with image embeddings |
| [`04_gaussian`](examples/04_gaussian) | Gaussian inference |
| [`05_linear_regression`](examples/05_linear_regression) | Linear regression |

## Documentation

For tutorials, configuration reference, and API docs, see **[cweniger.github.io/falcon](https://cweniger.github.io/falcon/)**.

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
