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

All `Flow` parameters are specified **flat** directly under `estimator:` in YAML.
There are no nested group keys — everything is a top-level keyword argument to `Flow.__init__`.

```yaml
estimator:
  _target_: falcon.estimators.Flow
  max_epochs: 300
  net_type: nsf
  lr: 0.01
  gamma: 0.5
  embedding:
    _target_: model.MyEmbedding
    _input_: [x]
```

## Configuration Reference

### Training Loop

Controls the training process.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_epochs` | int | 100 | Maximum training epochs |
| `batch_size` | int | 128 | Training batch size |
| `early_stop_patience` | int | 16 | Epochs without improvement before stopping |
| `cache_sync_every` | int | 0 | Epochs between cache syncs with the buffer (0 = every epoch) |
| `max_cache_samples` | int | 0 | Maximum samples to cache (0 = cache all available) |
| `cache_on_device` | bool | false | Keep cached training data on the estimator's device (e.g. GPU) |
| `prior_epochs` | int | 0 | Epochs to sample from prior before switching to proposal |
| `device` | str | null | Device string (e.g. `"cuda:0"`); auto-detected if `null` |

#### Data Caching

Training data is loaded into a local cache that is periodically synced with the shared simulation buffer. This avoids repeated remote data fetches and allows fast random-access batching.

- **`cache_sync_every`**: Controls how often the cache pulls new samples from the buffer. A value of `0` (default) syncs every epoch. Higher values reduce sync overhead at the cost of slightly stale data, which can be useful when simulations are slow.
- **`max_cache_samples`**: Caps the number of samples held in the cache. Set to `0` to cache everything. A positive value randomly subsamples, which helps limit GPU memory usage for very large buffers.
- **`cache_on_device`**: When `true`, cached tensors are moved to the estimator's device (typically GPU) once during sync rather than per-batch. This eliminates CPU-to-GPU transfer overhead during training but increases device memory usage.

### Network Architecture

Defines the neural network structure.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `net_type` | str | `zuko_nice` | Flow architecture (see [FlowDensity](flow-density.md) for all types) |
| `theta_norm` | bool | true | Normalize parameter space |
| `norm_momentum` | float | 0.01 | Momentum for online normalization updates |
| `use_log_update` | bool | false | Use log-space variance updates |
| `adaptive_momentum` | bool | false | Sample-dependent momentum |

### Embedding

The embedding network processes observations before they enter the flow.

```yaml
embedding:
  _target_: model.MyEmbedding
  _input_: [x]
```

See [Embeddings](embeddings.md) for details on the declarative embedding system, including multi-input and nested pipeline configurations.

### Optimizer

Controls learning rate and scheduling.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `lr` | float | 0.01 | Initial learning rate |
| `lr_decay_factor` | float | 0.1 | LR multiplier when plateau detected |
| `lr_patience` | int | 8 | Epochs without improvement before LR decay |
| `betas` | tuple | (0.9, 0.9) | AdamW beta coefficients |

### Inference

Controls posterior sampling and amortization.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `gamma` | float | 0.5 | Amortization mixing coefficient (0=focused, 1=amortized) |
| `discard_samples` | bool | true | Discard low-likelihood samples during training |
| `log_ratio_threshold` | float | -20 | Log-likelihood threshold for sample discarding |
| `sample_reference_posterior` | bool | false | Sample from reference posterior |
| `use_best_models` | bool | true | Use best validation model for sampling |
| `num_proposals` | int | 256 | Candidate samples drawn from the flow for importance sampling |
| `reference_samples` | int | 128 | Samples used to evaluate the reference posterior |
| `hypercube_bound` | float | 2.0 | Out-of-bounds threshold in hypercube space |
| `out_of_bounds_penalty` | float | 100.0 | Log-weight penalty applied to out-of-bounds proposals |
| `nan_replacement` | float | -100.0 | Log-weight substituted for NaN values during importance sampling |

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
      _target_: falcon.priors.Product
      priors:
        - ['uniform', -100.0, 100.0]
        - ['uniform', -100.0, 100.0]
        - ['uniform', -100.0, 100.0]

    estimator:
      _target_: falcon.estimators.Flow
      max_epochs: 100
      batch_size: 128
      early_stop_patience: 16
      cache_sync_every: 0
      max_cache_samples: 0
      net_type: zuko_nice
      theta_norm: true
      norm_momentum: 0.01
      embedding:
        _target_: model.E
        _input_: [x]
      lr: 0.01
      lr_decay_factor: 0.1
      lr_patience: 8
      gamma: 0.5
      discard_samples: true
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

Default configuration with continuous simulation:

```yaml
buffer:
  min_samples: 4096
  max_samples: 32768
  simulate_count: 128
  simulate_when_full: true
  simulate_interval: 10
```

### Amortized Training

Fixed dataset without simulation (for learning across many observations):

```yaml
buffer:
  min_samples: 32000
  max_samples: 32000
  simulate_count: 0       # No simulation
  simulate_when_full: false

graph:
  z:
    estimator:
      _target_: falcon.estimators.Flow
      gamma: 0.8          # Higher gamma for amortization
```

### Round-Based Training

Large batch renewal for sequential refinement:

```yaml
buffer:
  min_samples: 8000
  max_samples: 8000
  simulate_count: 8000    # Full renewal
  simulate_when_full: true
  simulate_interval: 30

graph:
  z:
    estimator:
      _target_: falcon.estimators.Flow
      discard_samples: true   # Remove poor samples
```

## Logged Metrics

Flow logs the following metrics during training:

| Metric | Description |
|--------|-------------|
| `train:loss` | Training loss (negative log-likelihood) |
| `val:loss` | Validation loss |
| `lr` | Current learning rate |
| `epoch` | Training epoch |
| `checkpoint:conditional` | Epoch when conditional flow was checkpointed |
| `checkpoint:marginal` | Epoch when marginal flow was checkpointed |

## Tips

1. **Start with defaults**: The default configuration works well for most problems
2. **Increase `max_epochs`** for complex posteriors
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

::: falcon.estimators.stepwise_base.TrainingLoopConfig
