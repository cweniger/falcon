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
  import_path: "."
  buffer_dir: "sim_dir"
  graph_dir: "graph_dir"
  samples_dir: "samples_dir"
```

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `import_path` | str | `"."` | Path to import custom modules |
| `buffer_dir` | str | `"sim_dir"` | Simulation buffer directory |
| `graph_dir` | str | `"graph_dir"` | Trained models directory |
| `samples_dir` | str | `"samples_dir"` | Output samples directory |

### `buffer`

Configure sample management:

```yaml
buffer:
  num_epochs: 500
  min_total_samples: 1000
  max_total_samples: 50000
  resample_interval: 10
  dump_interval: 50
```

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `num_epochs` | int | `500` | Total training epochs |
| `min_total_samples` | int | `1000` | Minimum samples before training |
| `max_total_samples` | int | `50000` | Maximum samples in buffer |
| `resample_interval` | int | `10` | Epochs between resampling |
| `dump_interval` | int | `50` | Epochs between buffer dumps |

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
      _target_: falcon.contrib.SNPE_A
      loop:
        num_epochs: 300
      network:
        net_type: nsf
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
  _target_: falcon.contrib.HypercubeMappingPrior
  priors:
    - ['uniform', -10.0, 10.0]
    - ['normal', 0.0, 1.0]
```

### `estimator`

The posterior learner (typically SNPE_A):

```yaml
estimator:
  _target_: falcon.contrib.SNPE_A

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
