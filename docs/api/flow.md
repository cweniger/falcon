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
- Round-based training with automatic learning rate scheduling and early stopping
  (see [Training Loop](../training.md))

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

Controls the training process. Training runs in rounds of epochs on fixed data; see
[Training Loop](../training.md) for the terminology and the full description.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_rounds` | int | null | Maximum number of rounds (`null` = unlimited) |
| `patience_rounds` | int | 10 | Stop training after this many rejected rounds in a row |
| `max_epochs` | int | 100 | Maximum epochs per round |
| `patience_epochs` | int | 16 | End the round once the best validation loss is this many epochs old |
| `val_every_epochs` | int | 1 | Validate after every N-th epoch (and at the last epoch of a round) |
| `batch_size` | int | 128 | Training batch size |
| `max_cache_samples` | int | 0 | Maximum samples to cache (0 = cache all available) |
| `cache_on_device` | bool | false | Keep cached training data on the estimator's device (e.g. GPU) |
| `prior_rounds` | int | 0 | Rounds that simulate from the prior before switching to the learned proposal (the first round always does) |
| `device` | str | null | Device string (e.g. `"cuda:0"`); auto-detected if `null` |

#### Data Caching

Training data is loaded into a local cache that is refreshed from the shared simulation buffer at the start of every round and then stays fixed for the round. This avoids repeated remote data fetches and allows fast batching.

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
| `lr` | float | 0.01 | Learning rate at the start of every round |
| `lr_decay_factor` | float | 1.0 | LR multiplier when plateau detected (1.0 = no decay) |
| `lr_patience_epochs` | int | 8 | Epochs without validation improvement before LR decay |
| `betas` | tuple | (0.9, 0.9) | AdamW beta coefficients |

### Inference

Controls posterior sampling and amortization.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `gamma` | float | 0.5 | Amortization mixing coefficient (0=focused, 1=amortized) |
| `discard_samples` | bool | true | After each accepted round, discard low-likelihood samples |
| `log_ratio_threshold` | float | -20 | Log-likelihood threshold for sample discarding |
| `sample_reference_posterior` | bool | false | Sample from reference posterior |
| `use_best_models` | bool | true | Sample from the best networks (instead of the networks being trained) |
| `num_proposals` | int | 256 | Candidate samples drawn from the flow for importance sampling |
| `reference_samples` | int | 128 | Samples used to evaluate the reference posterior |
| `hypercube_bound` | float | 2.0 | Out-of-bounds threshold in hypercube space |
| `out_of_bounds_penalty` | float | 100.0 | Log-weight penalty applied to out-of-bounds proposals |
| `nan_replacement` | float | -100.0 | Log-weight substituted for NaN values during importance sampling |

#### Understanding `gamma`

`gamma` controls how broadly the proposal distribution covers the parameter space:

- **`gamma=0`**: Proposal equals the posterior — tightest possible proposal, best for refining a good posterior estimate
- **Higher `gamma`**: Broader proposal — more exploration, more robust to poorly-initialized posteriors
- **`gamma=0.5`**: Default, works well in most cases

Note: `gamma` is not the same as "amortization" in the NPE literature. It sets the de-weighting of the conditional flow relative to the marginal during importance sampling.

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
      patience_epochs: 16
      patience_rounds: 10
      batch_size: 128
      max_cache_samples: 0
      net_type: zuko_nice
      theta_norm: true
      norm_momentum: 0.01
      embedding:
        _target_: model.E
        _input_: [x]
      lr: 0.01
      lr_decay_factor: 1.0
      lr_patience_epochs: 8
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
| `train:loss` | Training loss of the conditional flow (negative log-likelihood) |
| `train:loss_aux` | Training loss of the marginal flow |
| `val:loss`, `val:loss_aux` | Validation losses (only for validated epochs) |
| `lr` | Current learning rate |
| `round`, `epoch` | Current round, and epoch within the round |
| `checkpoint:conditional` | Epoch at which the conditional flow reached its best validation loss in the round |
| `checkpoint:marginal` | Epoch at which the marginal flow reached its best validation loss in the round |
| `round:accepted` | 1 if the round was accepted |
| `round:conditional:promoted`, `round:marginal:promoted` | 1 if the flow replaced its best network |

See [Training Loop](../training.md#log-output-and-metrics) for all round metrics.

## Tips

1. **Start with defaults**: The default configuration works well for most problems
2. **Increase `max_epochs`** (per round) for complex posteriors
3. **Enable `discard_samples`** if training becomes unstable with outliers
4. **Use GPU** (`ray.num_gpus: 1`) for faster training with large embeddings
5. **Lower `gamma`** for single-observation inference, higher for amortization
6. **Adjust `patience_epochs`** based on expected convergence time within a round
7. **Set `cache_on_device: true`** when GPU memory permits, to eliminate per-batch CPU-to-GPU transfers
8. **Increase `val_every_epochs`** (e.g. 3) when validation is expensive; `patience_epochs` keeps its meaning

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

## Training Loop

`Flow` inherits its round loop from `StepwiseEstimator`. The loop parameters
(`max_rounds`, `patience_rounds`, `max_epochs`, `patience_epochs`,
`val_every_epochs`, `batch_size`, `max_cache_samples`, `cache_on_device`,
`prior_rounds`) are `Flow.__init__` arguments, documented above.

::: falcon.estimators.stepwise_base.StepwiseEstimator
    options:
      show_source: false
      members:
        - train
