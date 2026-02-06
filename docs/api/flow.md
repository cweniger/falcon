# Flow Estimator

Flow-based posterior estimation using normalizing flows.

## Overview

`Flow` is the primary estimator in Falcon for learning posterior distributions.
It uses dual normalizing flows (conditional and marginal) with importance sampling
for adaptive proposal generation.

Key features:

- Dual flow architecture for posterior and proposal sampling
- Parameter space normalization via hypercube mapping
- Importance sampling with effective sample size monitoring
- Automatic learning rate scheduling and early stopping

## Configuration

Flow is configured through four groups: `loop`, `network`, `optimizer`, and `inference`.
The `embedding` configuration sits as a sibling of `network`, not nested inside it.

```yaml
estimator:
  _target_: falcon.estimators.Flow

  loop:
    # Training loop parameters

  network:
    # Neural network architecture

  embedding:
    # Observation embedding network

  optimizer:
    # Learning rate and scheduling

  inference:
    # Sampling and amortization settings
```

## Configuration Reference

### Training Loop (`loop`)

Controls the training process.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_epochs` | int | 300 | Maximum training epochs |
| `batch_size` | int | 128 | Training batch size |
| `early_stop_patience` | int | 32 | Epochs without improvement before stopping |
| `reset_network_after_pause` | bool | false | Reset network weights when training resumes after pause |
| `cache_sync_every` | int | 0 | Epochs between cache syncs with the buffer (0 = every epoch) |
| `max_cache_samples` | int | 0 | Maximum samples to cache (0 = cache all available) |
| `cache_on_device` | bool | false | Keep cached training data on the estimator's device (e.g. GPU) |

```yaml
loop:
  num_epochs: 300
  batch_size: 128
  early_stop_patience: 32
  reset_network_after_pause: false
  cache_sync_every: 0
  max_cache_samples: 0
  cache_on_device: false
```

#### Data Caching

Training data is loaded into a local cache that is periodically synced with the shared simulation buffer. This avoids repeated remote data fetches and allows fast random-access batching.

- **`cache_sync_every`**: Controls how often the cache pulls new samples from the buffer. A value of `0` (default) syncs every epoch. Higher values reduce sync overhead at the cost of slightly stale data, which can be useful when simulations are slow.
- **`max_cache_samples`**: Caps the number of samples held in the cache. Set to `0` to cache everything. A positive value randomly subsamples, which helps limit GPU memory usage for very large buffers.
- **`cache_on_device`**: When `true`, cached tensors are moved to the estimator's device (typically GPU) once during sync rather than per-batch. This eliminates CPU-to-GPU transfer overhead during training but increases device memory usage.

### Network Architecture (`network`)

Defines the neural network structure.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `net_type` | str | `nsf` | Flow architecture (see [FlowDensity](flow-density.md) for all types) |
| `theta_norm` | bool | true | Normalize parameter space |
| `norm_momentum` | float | 0.003 | Momentum for online normalization updates |
| `use_log_update` | bool | false | Use log-space variance updates |
| `adaptive_momentum` | bool | false | Sample-dependent momentum |

```yaml
network:
  net_type: nsf
  theta_norm: true
  norm_momentum: 0.003
  use_log_update: false
  adaptive_momentum: false
```

### Embedding

The embedding network processes observations before they enter the flow. It is configured as a **sibling** of `network` (not nested inside it).

```yaml
embedding:
  _target_: model.MyEmbedding
  _input_: [x]
```

See [Embeddings](embeddings.md) for details on the declarative embedding system, including multi-input and nested pipeline configurations.

### Optimizer (`optimizer`)

Controls learning rate and scheduling.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `lr` | float | 0.01 | Initial learning rate |
| `lr_decay_factor` | float | 0.5 | LR multiplier when plateau detected |
| `scheduler_patience` | int | 16 | Epochs without improvement before LR decay |

```yaml
optimizer:
  lr: 0.01
  lr_decay_factor: 0.5
  scheduler_patience: 16
```

### Inference (`inference`)

Controls posterior sampling and amortization.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `gamma` | float | 0.5 | Amortization mixing coefficient (0=focused, 1=amortized) |
| `discard_samples` | bool | false | Discard low-likelihood samples during training |
| `log_ratio_threshold` | float | -20 | Log-likelihood threshold for sample discarding |
| `sample_reference_posterior` | bool | false | Sample from reference posterior |
| `use_best_models_during_inference` | bool | true | Use best validation model for sampling |

```yaml
inference:
  gamma: 0.5
  discard_samples: false
  log_ratio_threshold: -20
  sample_reference_posterior: false
  use_best_models_during_inference: true
```

#### Understanding `gamma` (Amortization)

The `gamma` parameter controls the trade-off between focused and amortized inference:

- **`gamma=0`**: Fully focused on the specific observation (best for single-observation inference)
- **`gamma=1`**: Fully amortized (network generalizes across observations)
- **`gamma=0.5`**: Balanced (default, good for most cases)

## Embedding Networks

Flow requires an embedding network to process observations. The embedding maps high-dimensional observations to a lower-dimensional summary statistic.

### Basic Embedding

```yaml
embedding:
  _target_: model.MyEmbedding
  _input_: [x]
```

### Multi-Input Embedding

```yaml
embedding:
  _target_: model.MyEmbedding
  _input_: [x, y]  # Multiple observation nodes
```

### Nested Embedding Pipeline

```yaml
embedding:
  _target_: model.Concatenate
  _input_:
    - _target_: timm.create_model
      model_name: resnet18
      pretrained: true
      num_classes: 0
      _input_:
        _target_: model.Unsqueeze
        _input_: [image]
    - _target_: torch.nn.Linear
      in_features: 64
      out_features: 32
      _input_: [metadata]
```

## Complete Example

```yaml
graph:
  z:
    evidence: [x]

    simulator:
      _target_: falcon.priors.Hypercube
      priors:
        - ['uniform', -100.0, 100.0]
        - ['uniform', -100.0, 100.0]
        - ['uniform', -100.0, 100.0]

    estimator:
      _target_: falcon.estimators.Flow

      loop:
        num_epochs: 300
        batch_size: 128
        early_stop_patience: 32
        cache_sync_every: 0
        max_cache_samples: 0

      network:
        net_type: nsf
        theta_norm: true
        norm_momentum: 0.003

      embedding:
        _target_: model.E
        _input_: [x]

      optimizer:
        lr: 0.01
        lr_decay_factor: 0.5
        scheduler_patience: 16

      inference:
        gamma: 0.5
        discard_samples: false
        log_ratio_threshold: -20

    ray:
      num_gpus: 0

  x:
    parents: [z]
    simulator:
      _target_: model.Simulate
    observed: "./data/obs.npz['x']"
```

## Training Strategies

### Standard Training

Default configuration with continuous resampling:

```yaml
buffer:
  min_training_samples: 4096
  max_training_samples: 32768
  resample_batch_size: 128
  keep_resampling: true
  resample_interval: 10
```

### Amortized Training

Fixed dataset without resampling (for learning across many observations):

```yaml
buffer:
  min_training_samples: 32000
  max_training_samples: 32000
  resample_batch_size: 0       # No resampling
  keep_resampling: false

# Higher gamma for amortization
inference:
  gamma: 0.8
```

### Round-Based Training

Large batch renewal for sequential refinement:

```yaml
buffer:
  min_training_samples: 8000
  max_training_samples: 8000
  resample_batch_size: 8000    # Full renewal
  keep_resampling: true
  resample_interval: 30        # Less frequent

inference:
  discard_samples: true        # Remove poor samples
```

## Logged Metrics

Flow logs the following metrics during training:

| Metric | Description |
|--------|-------------|
| `loss/train` | Training loss (negative log-likelihood) |
| `loss/val` | Validation loss |
| `lr` | Current learning rate |
| `epoch` | Training epoch |
| `best_val_loss` | Best validation loss seen |

## Tips

1. **Start with defaults**: The default configuration works well for most problems
2. **Increase `num_epochs`** for complex posteriors
3. **Enable `discard_samples`** if training becomes unstable with outliers
4. **Use GPU** (`ray.num_gpus: 1`) for faster training with large embeddings
5. **Lower `gamma`** for single-observation inference, higher for amortization
6. **Adjust `early_stop_patience`** based on expected convergence time
7. **Set `cache_on_device: true`** when GPU memory permits, to eliminate per-batch CPU-to-GPU transfers
8. **Increase `cache_sync_every`** (e.g. 5-10) when simulations are slow and training data changes infrequently

## Class Reference

::: falcon.estimators.flow.Flow
    options:
      show_source: true
      members:
        - __init__
        - train_step
        - val_step
        - sample_prior
        - sample_posterior
        - sample_proposal
        - save
        - load

## Configuration Classes

::: falcon.estimators.flow.FlowConfig

::: falcon.estimators.flow.NetworkConfig

::: falcon.estimators.flow.OptimizerConfig

::: falcon.estimators.flow.InferenceConfig

::: falcon.estimators.base.TrainingLoopConfig
