# FALCON - Federated Adaptive Learning of CONditional distributions

FALCON is a Python framework for **simulation-based inference (SBI)** that enables adaptive learning of complex conditional distributions. Built on top of PyTorch, Ray, and sbi, FALCON provides a declarative approach to building probabilistic models with automatic parallelization and experiment tracking.

## Key Features

- **Declarative Model Definition**: Define complex probabilistic models using simple YAML configuration files
- **Adaptive Sampling**: Automatically adjusts simulation parameters based on training progress
- **Distributed Computing**: Built-in support for parallel execution using Ray
- **Neural Density Estimation**: Multiple neural network architectures for posterior estimation (NSF, MAF, NAF, etc.)
- **Experiment Tracking**: Integrated Weights & Biases (WandB) logging for monitoring training
- **Flexible Architecture**: Modular design supporting various SBI algorithms including SNPE-A

## Installation

### Prerequisites

- Python 3.8 or higher
- PyTorch 2.0 or higher
- CUDA (optional, for GPU acceleration)

### Install from source

```bash
# Clone the repository
git clone https://github.com/yourusername/falcon.git
cd falcon

# Install in development mode
pip install -e .
```

### Dependencies

FALCON requires the following packages (automatically installed):
- `torch>=2.0.0` - Deep learning framework
- `numpy` - Numerical computing
- `ray` - Distributed computing
- `sbi` - Simulation-based inference toolbox
- `hydra-core` - Configuration management
- `omegaconf` - Configuration utilities
- `wandb>=0.15.0` - Experiment tracking

## Project Structure

```
falcon/
├── falcon/                 # Core library
│   ├── core/              # Core components
│   │   ├── graph.py       # Graph-based model definition
│   │   ├── deployed_graph.py  # Runtime graph execution
│   │   ├── raystore.py    # Distributed data management
│   │   └── logging.py     # Logging utilities
│   ├── contrib/           # Built-in components
│   │   ├── SNPE_A.py      # SNPE-A implementation
│   │   ├── torch_embedding.py  # Neural embeddings
│   │   └── hypercubemappingprior.py  # Prior distributions
│   └── cli.py             # Command-line interface
├── examples/              # Example configurations
│   ├── 01_minimal/        # Basic SBI example
│   └── 02_bimodal/        # Advanced bimodal example
└── setup.py               # Package configuration
```

## Configuration Structure

FALCON uses Hydra/OmegaConf for configuration management. Each experiment is defined by a `config.yaml` file:

### Core Configuration Sections

```yaml
# Experiment tracking with Weights & Biases
logging:
  project: falcon_examples       # WandB project name
  group: experiment_01           # Experiment group
  dir: ${hydra:run.dir}         # Output directory

# Directory paths
paths:
  import: "./src"               # User code location
  buffer: ${hydra:run.dir}/sim_dir    # Simulation data
  graph: ${hydra:run.dir}/graph_dir   # Trained models

# Training buffer configuration
buffer:
  min_training_samples: 4096    # Min samples before training
  max_training_samples: 32768   # Max buffer size
  validation_window_size: 256   # Validation split size
  resample_batch_size: 128      # New samples per iteration
  keep_resampling: true         # Continue after max reached
  resample_interval: 10         # Epochs between resampling

# Graph definition (model architecture)
graph:
  # Define nodes and their relationships
  parameter_node:
    evidence: [observation_node]  # Inference target
    simulator:                   # Prior distribution
      _target_: falcon.contrib.HypercubeMappingPrior
      priors: [['uniform', -10, 10], ...]
    estimator:                   # Posterior estimator
      _target_: falcon.contrib.SNPE_A
      net_type: nsf              # Neural network type
      num_epochs: 300
      batch_size: 128
      lr: 0.01
    ray:
      num_gpus: 0                # GPU allocation

  observation_node:
    parents: [parameter_node]    # Dependencies
    simulator:                   # Forward model
      _target_: model.YourSimulator
    observed: "./data/obs.npy"   # Observed data

# Sampling configuration
sample:
  posterior:
    n: 1000                      # Number of samples
    path: samples_posterior.joblib
```

### Key Configuration Parameters

#### Buffer Settings
- `min_training_samples`: Minimum samples required before training begins
- `max_training_samples`: Maximum number of samples retained in memory
- `keep_resampling`: Whether to continue generating samples after reaching maximum
- `resample_interval`: How often (in epochs) to generate new samples

#### Estimator Options
- `net_type`: Neural network architecture (`nsf`, `maf`, `naf`, `gf`, `zuko_gf`)
- `gamma`: SNPE-A mixing coefficient for amortization
- `theta_norm`: Enable parameter space normalization
- `early_stop_patience`: Epochs without improvement before stopping

#### Ray Configuration
- `num_gpus`: GPU count per worker (0 for CPU-only)
- `init`: Additional Ray initialization parameters

## Usage

FALCON provides a simple command-line interface for running experiments:

### Launch Training

```bash
# Run training with default configuration
falcon launch

# Run with specific output directory
falcon launch hydra.run.dir=outputs/my_experiment

# Override configuration parameters
falcon launch buffer.num_epochs=500 graph.z.estimator.lr=0.001
```

### Generate Samples

```bash
# Sample from prior distribution
falcon sample prior

# Sample from trained posterior (after training)
falcon sample posterior

# Sample from proposal distribution
falcon sample proposal
```

### Running Examples

#### Example 1: Minimal Configuration

```bash
cd examples/01_minimal
falcon launch hydra.run.dir=outputs/run_01
falcon sample posterior hydra.run.dir=outputs/run_01
```

This example demonstrates:
- Basic 3-parameter inference problem
- Simple forward model
- Neural spline flow (NSF) posterior estimation

#### Example 2: Bimodal Distribution

```bash
cd examples/02_bimodal
# Regular training
falcon launch --config-name config_regular

# Amortized inference
falcon launch --config-name config_amortized

# Round-based training with renewal
falcon launch --config-name config_rounds_renew
```

This example showcases:
- Complex 10-dimensional parameter space
- Bimodal posterior distributions
- Different training strategies (regular, amortized, round-based)
- GPU acceleration

## Creating Your Own Models

### Step 1: Define Your Simulator

Create a Python module in your `src/` directory:

```python
# src/model.py
import torch

class MySimulator:
    def __init__(self, param1=1.0):
        self.param1 = param1

    def __call__(self, z):
        # z: input parameters [batch_size, n_params]
        # return: simulated data [batch_size, n_observables]
        return torch.sin(z * self.param1)

class MyEmbedding(torch.nn.Module):
    def __init__(self, input_dim, output_dim=32):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.net(x)
```

### Step 2: Create Configuration

```yaml
# config.yaml
logging:
  project: my_project
  group: experiment_01

paths:
  import: "./src"
  buffer: ${hydra:run.dir}/sim_dir
  graph: ${hydra:run.dir}/graph_dir

buffer:
  min_training_samples: 1024
  max_training_samples: 8192

graph:
  theta:
    evidence: [x]
    simulator:
      _target_: falcon.contrib.HypercubeMappingPrior
      priors: [['uniform', -5, 5], ['uniform', -5, 5]]
    estimator:
      _target_: falcon.contrib.SNPE_A
      embedding:
        _target_: model.MyEmbedding
        _input_: [x]
        input_dim: 10
        output_dim: 32
      net_type: nsf
      num_epochs: 200

  x:
    parents: [theta]
    simulator:
      _target_: model.MySimulator
      param1: 2.0
    observed: "./data/observations.npy"
```

### Step 3: Prepare Observations

```python
# Generate or load your observed data
import numpy as np
observations = np.random.randn(10)  # Your actual data
np.save("data/observations.npy", observations)
```

### Step 4: Run Training

```bash
falcon launch
```

## Advanced Features

### Multi-Node Graphs

FALCON supports complex dependency graphs with multiple nodes:

```yaml
graph:
  # Latent parameters
  z:
    evidence: [x]
    simulator: ...
    estimator: ...

  # Intermediate signal
  signal:
    parents: [z]
    simulator: model.SignalGenerator

  # Noise component
  noise:
    simulator: model.NoiseGenerator

  # Final observation
  x:
    parents: [signal, noise]
    simulator: model.Combiner
    observed: "./data/obs.npy"
```

### Custom Neural Networks

Implement custom embedding networks for complex data:

```python
class ConvolutionalEmbedding(torch.nn.Module):
    """For image data"""
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, 3),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Flatten()
        )

    def forward(self, x):
        return self.conv(x)
```

### Distributed Training

Configure Ray for distributed execution:

```yaml
ray:
  init:
    num_cpus: 8
    num_gpus: 2
    dashboard_host: "0.0.0.0"

graph:
  z:
    ray:
      num_gpus: 1  # Per-worker GPU allocation
```

## Monitoring and Debugging

### Weights & Biases Integration

FALCON automatically logs training metrics to WandB:

1. Set up WandB account: https://wandb.ai
2. Configure project in `config.yaml`
3. Monitor training at https://wandb.ai/your-username/your-project

Logged metrics include:
- Training/validation loss
- Learning rate schedules
- Sample generation statistics
- Model architecture details

### Output Structure

After training, FALCON creates:

```
outputs/run_name/
├── sim_dir/           # Generated simulation data
│   └── samples_*.pt   # Training samples
├── graph_dir/         # Trained models
│   ├── graph.pkl      # Graph structure
│   └── estimators/    # Neural network weights
├── samples_posterior.joblib  # Posterior samples
└── .hydra/            # Configuration logs
```

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce `buffer.max_training_samples` or `batch_size`
2. **Slow Training**: Enable GPU with `ray.num_gpus: 1`
3. **Poor Convergence**: Adjust `lr`, `num_epochs`, or try different `net_type`
4. **Import Errors**: Ensure `paths.import` points to your model code

### Performance Tips

- Use GPU acceleration for large models
- Enable `theta_norm` for high-dimensional parameter spaces
- Adjust `resample_interval` based on simulation cost
- Use `early_stop_patience` to prevent overfitting

## Citation

If you use FALCON in your research, please cite:

```bibtex
@software{falcon2024,
  title = {FALCON: Federated Adaptive Learning of CONditional distributions},
  year = {2024},
  url = {https://github.com/yourusername/falcon}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Support

For questions and support, please open an issue on GitHub or contact the maintainers.