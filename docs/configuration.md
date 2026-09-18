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
  imports: ["."]
  graph:   ${run_dir}/graph
  samples: ${run_dir}/samples
  buffer:  ${run_dir}/buffer   # optional; redirect to a separate volume (e.g. scratch)
```

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `imports` | list[str] | `null` | Directories prepended to `sys.path` in Ray workers so custom modules (e.g. `model.Simulator`) can be imported |
| `graph` | str | `${run_dir}/graph` | Trained model checkpoints directory |
| `samples` | str | `${run_dir}/samples` | Output samples directory |
| `buffer` | str | `${run_dir}/buffer` | Buffer snapshots directory (`snapshots/` is appended); useful for routing large temporary simulation data to a separate scratch volume while keeping `run_dir` on persistent storage |

### `buffer`

Configure the rolling sample buffer that feeds training. Falcon continuously simulates new samples in the background while training runs concurrently.

```yaml
buffer:
  min_samples: 4096
  max_samples: 32768
  validation_fraction: 0.15
  simulate_count: 64
  simulate_interval: 1
  simulate_when_full: true
  snapshot_every: 0
```

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `min_samples` | int | — | Minimum number of samples (training + validation) required before training starts |
| `max_samples` | int | — | Maximum number of samples (training + validation) retained; the oldest samples are permanently removed when this is exceeded |
| `validation_fraction` | float | `0.15` | Share of samples held out for validation only (early stopping, LR schedule, best checkpoint). Assigned by insertion order and spread evenly over the buffer; a sample's role never changes, and validation samples age out and are discarded under the same rules as training samples. Must lie in (0, 0.5] |
| `simulate_count` | int | `64` | Number of new samples generated per simulation round. For simulators taking >1s per sample, keep this small (4–16) to avoid long delays between buffer updates; for fast simulators, increase to reduce Ray overhead. |
| `simulate_interval` | float | `1` | Seconds between simulation rounds |
| `simulate_when_full` | bool | `true` | If `true`, simulation continues after `max_samples` is reached and old samples are replaced; if `false`, simulation stops once the buffer is full |
| `snapshot_every` | int | `0` | Save every Nth sample to `{paths.buffer}/snapshots/` for inspection (0 = disabled, 1 = all, 10 = every 10th sample) |
| `simulate_chunk_size` | int | `0` | Max samples per individual simulation call (0 = full `simulate_count` in one call) |
| `initial_samples_path` | str | `null` | Path to a pre-existing sample type directory to pre-load into the buffer on startup |

### `graph`

Define the computational graph. Each key is a node name:

```yaml
graph:
  node_name:
    parents: [parent1, parent2]    # Forward model dependencies
    evidence: [evidence1]          # Inference dependencies (drive backward traversal)
    scaffolds: [scaffold1]         # Extra conditioning inputs (passed to estimator but not inferred)
    observed: "./path/to/data.npz" # Observation file
    resample: false                # If true, re-draw samples from proposal each round instead of accumulating

    simulator:                     # Forward model
      _target_: module.ClassName
      param1: value1

    estimator:                     # Posterior learner (optional)
      _target_: falcon.estimators.Flow
      max_epochs: 300
      net_type: nsf
      embedding:
        _target_: model.MyEmbedding
        _input_: [x]
      lr: 0.01
      gamma: 0.5

    ray:                          # Ray actor configuration
      num_train_gpus: 0.5         # GPUs of the train actor (estimator nodes only)
      num_sample_gpus: 0.5        # GPUs of the sample actor
      num_cpus: 1
```

## Node Configuration

### `simulator`

The forward model that generates samples:

```yaml
simulator:
  _target_: falcon.priors.Product
  priors:
    - ['uniform', -10.0, 10.0]
    - ['normal', 0.0, 1.0]
```

### `estimator`

The posterior learner. Falcon provides three estimators:

- [`falcon.estimators.Flow`](api/flow.md) — Flow-based posterior estimation (recommended for most cases)
- [`falcon.estimators.FlowMatching`](api/flow-matching.md) — Flow matching with a truncated-prior proposal
- [`falcon.estimators.GaussianFullCov`](api/gaussian.md) — Full covariance Gaussian posterior

All estimator parameters are specified **flat** directly under `estimator:` — there
are no nested group keys (`loop`, `network`, etc.). The `embedding` key is special:
it takes a nested `_target_` / `_input_` block as usual. Estimators train in rounds
of epochs on fixed data; see [Training Loop](training.md) for the loop parameters.

```yaml
estimator:
  _target_: falcon.estimators.Flow
  max_epochs: 300        # per round
  patience_epochs: 50
  patience_rounds: 10
  batch_size: 128
  max_cache_samples: 0
  cache_on_device: false
  net_type: nsf          # nsf, maf, zuko_nice, etc.
  theta_norm: true
  embedding:
    _target_: model.Embedding
    _input_: [x]
  lr: 0.01
  lr_decay_factor: 1.0
  lr_patience_epochs: 8
  gamma: 0.5             # Proposal breadth (0=tight around posterior, higher=broader)
  discard_samples: true
```

### `ray`

Per-node Ray resource allocation. Every node has a **sample actor** (named
after the node, e.g. `z`) that serves prior, proposal and posterior samples.
Nodes with an estimator also have a **train actor** (`z/train`) that trains
the networks and sends each new best network to the sample actor, so sampling
and training run at the same time (see [Training Loop](training.md#two-actors-per-estimator-node)).

```yaml
ray:
  num_train_gpus: 0.5
  num_sample_gpus: 0.5
  num_cpus: 2
```

| Key | Default | Description |
|-----|---------|-------------|
| `num_train_gpus` | `0` | GPUs of the train actor; ignored for nodes without an estimator |
| `num_sample_gpus` | `0` | GPUs of each sample actor |
| `num_gpus` | — | Deprecated. On estimator nodes it is split evenly between the two actors (`num_gpus: 1` = 0.5 + 0.5); otherwise it sets `num_sample_gpus`. Cannot be combined with the keys above |
| `num_train_cpus` | `num_cpus` | CPUs of the train actor; also its number of CPU threads |
| `num_sample_cpus` | `num_cpus` | CPUs of each sample actor; also its number of CPU threads |
| other Ray actor options | — | `num_cpus`, `memory`, `resources`, `runtime_env`, … apply to each actor of the node |

Training and sampling compute at the same time, so they must not both use all
CPU cores. An actor with CPUs set uses that many threads. Otherwise the
estimator actors split this machine's cores evenly (with one estimator node:
half each for training and sampling); nodes without an estimator keep their
library defaults. The thread counts are shown when the graph starts.

Both actors may share one GPU with fractional values. Ray does not limit the
memory of fractional GPUs, so leave headroom for two processes. Leave the
estimator's `device` unset so that each actor picks its own device; an
explicit `device: cuda` fails in an actor without a GPU.

`num_actors: N` on a node starts N sample actors that share the sampling work;
an estimator node still has a single train actor, which updates all of them.

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

## `sample`

Configure automatic post-training sampling. Each key under `sample` matches a
sample type (`prior`, `posterior`, `proposal`, `ppd`):

```yaml
sample:
  posterior:
    n: 1000          # Number of samples to draw
  ppd:
    n: 500           # Posterior predictive samples (requires observed nodes with parents)
```

If `n > 0`, sampling runs automatically after training completes. Samples are
written to `{paths.samples}/{type}/`.

## Overriding Configuration

Override any parameter via CLI:

```bash
falcon launch buffer.max_samples=32768 graph.theta.estimator.lr=0.001
```
