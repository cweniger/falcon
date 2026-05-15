# Configuration

Falcon uses YAML configuration files powered by OmegaConf. This page documents all available options.

## Configuration Sections

### `logging`

Configure experiment tracking:

```yaml
logging:
  wandb:
    enabled: true
    project: "my-project"
    entity: "my-team"
  local:
    enabled: true
```

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `wandb.enabled` | bool | `false` | Enable WandB logging |
| `wandb.project` | str | `"falcon"` | WandB project name |
| `wandb.entity` | str | `null` | WandB team/entity |
| `local.enabled` | bool | `true` | Enable local file logging |

### `paths`

Configure file paths:

```yaml
paths:
  import: "."
  buffer: ${run_dir}/sim_dir
  graph: ${run_dir}/graph_dir
  samples: ${run_dir}/samples_dir
```

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `import` | str | `"."` | Path to import custom modules |
| `buffer` | str | `${run_dir}/sim_dir` | Simulation buffer directory |
| `graph` | str | `${run_dir}/graph_dir` | Trained models directory |
| `samples` | str | `${run_dir}/samples_dir` | Output samples directory |

### `buffer`

Configure the rolling sample buffer that feeds training. Falcon continuously simulates new samples in the background while training runs concurrently.

```yaml
buffer:
  min_samples: 4096
  max_samples: 32768
  validation_samples: 256
  simulate_count: 64
  simulate_interval: 1
  simulate_when_full: true
  store_fraction: 0.0
```

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `min_samples` | int | — | Minimum training samples required before training starts |
| `max_samples` | int | — | Maximum training samples retained; older samples are disfavoured once this is exceeded |
| `validation_samples` | int | — | Number of samples held out for validation (used for early stopping) |
| `simulate_count` | int | `64` | Number of new samples generated per simulation round. For simulators taking >1s per sample, keep this small (4–16) to avoid long delays between buffer updates; for fast simulators, increase to reduce Ray overhead. |
| `simulate_interval` | float | `1` | Seconds between simulation rounds |
| `simulate_when_full` | bool | `true` | If `true`, simulation continues after `max_samples` is reached and old samples are replaced; if `false`, simulation stops once the buffer is full |
| `store_fraction` | float | `0.0` | Fraction of simulated samples written to `samples_dir/buffer/` for inspection (0 = none, 1 = all) |

### `graph`

Define the computational graph. Each key is a node name:

```yaml
graph:
  node_name:
    parents: [parent1, parent2]    # Forward model dependencies
    evidence: [evidence1]          # Inference dependencies
    scaffolds: [scaffold1]         # Additional conditioning
    observed: "./path/to/data.npz" # Observation file
    resample: false                # Enable adaptive resampling

    simulator:                     # Forward model
      _target_: module.ClassName
      param1: value1

    estimator:                     # Posterior learner (optional)
      _target_: falcon.estimators.Flow
      loop:
        num_epochs: 300
      network:
        net_type: nsf
      embedding:
        _target_: model.MyEmbedding
        _input_: [x]
      optimizer:
        lr: 0.01
      inference:
        gamma: 0.5

    ray:                          # Ray actor configuration
      num_gpus: 0
      num_cpus: 1
```

## Node Configuration

### `simulator`

The forward model that generates samples:

```yaml
simulator:
  _target_: falcon.priors.Hypercube
  priors:
    - ['uniform', -10.0, 10.0]
    - ['normal', 0.0, 1.0]
```

### `estimator`

The posterior learner. Falcon provides two estimators:

- [`falcon.estimators.Flow`](api/flow.md) — Flow-based posterior estimation (recommended for most cases)
- [`falcon.estimators.Gaussian`](api/gaussian.md) — Full covariance Gaussian posterior

```yaml
estimator:
  _target_: falcon.estimators.Flow

  loop:
    num_epochs: 300
    batch_size: 128
    early_stop_patience: 50
    cache_sync_every: 0
    max_cache_samples: 0
    cache_on_device: false

  network:
    net_type: nsf          # nsf, maf, zuko_nice, etc.
    theta_norm: true

  embedding:
    _target_: model.Embedding
    _input_: [x]

  optimizer:
    lr: 0.01
    lr_decay_factor: 0.1
    scheduler_patience: 8

  inference:
    gamma: 0.5             # Proposal width (0=posterior, 1=prior)
    discard_samples: true
```

### `ray`

Per-node Ray resource allocation:

```yaml
ray:
  num_gpus: 1
  num_cpus: 2
```

## Global Ray Configuration

```yaml
ray:
  num_cpus: 8
  num_gpus: 1
  object_store_memory: 1000000000
```

## Observation Syntax

Load data from NPZ files with optional key extraction:

```yaml
# Single-key NPZ (auto-extracted)
observed: "./data/obs.npz"

# Specific key extraction
observed: "./data/obs.npz['x']"
```

## Overriding Configuration

Override any parameter via CLI:

```bash
falcon launch buffer.num_epochs=1000 graph.theta.estimator.optimizer.lr=0.001
```
