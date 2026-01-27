# Goal

Write a local logging system that complements the wandb logging system.

## Storage Format

**Chunked NPZ files per metric**, stored within the run directory:

```
graph/
  z/
    metrics/
      loss/
        chunk_000000.npz    # entries 0-99
        chunk_000100.npz    # entries 100-199
        ...
      lr/
        chunk_000000.npz
      embedding_loss/
        chunk_000000.npz
  theta/
    metrics/
      loss/
        chunk_000000.npz
```

Each NPZ file contains:
```python
np.savez(path, step=steps_array, value=values_array, walltime=walltime_array)
# step: int64 array
# value: float64 array
# walltime: float64 array (epoch time)
```

**Why NPZ over JSONL:**
- ~3x smaller storage (24 bytes/entry vs 65 bytes)
- Native numpy load (no parsing overhead)
- At scale (100 metrics × 10 nodes × 720k entries) = 17GB vs 47GB

## Write Behavior

- Each metric maintains its own in-memory buffer
- Flush to disk every 100 entries (configurable)
- Flush remaining buffer on logger shutdown
- Each actor/node gets its own logging directory → no file conflicts
- Append-friendly: chunks are write-once, never modified

## Read API

```python
run = falcon.read_run(path)
run['z']['loss'].values     # np.array, concatenated from all chunks
run['z']['loss'].steps      # np.array of step numbers
run['z']['loss'].walltime   # np.array of timestamps
```

- Lazy loading: chunks loaded on demand
- Extensible for future metadata

## Current API (must not change)

```python
falcon.log(metrics: dict, log_prefix=None)
```

Usage examples from codebase:
```python
falcon.log({"x_mean": x.mean().item()})
falcon.log({"Signal:mean": m.mean().item()})
falcon.log({f"{self.log_prefix}embed_min": y.min().item()})
```

Current architecture:
- `falcon/core/logging.py`: exposes `log()`, `initialize_logging_for(actor_id)`
- `falcon/core/wandb_logger.py`: `WandBManager` (Ray actor) manages per-actor `WandBWrapper` instances
- Metrics prefixed with `actor_id:log_prefix:key`
- Step currently `None` (wandb auto-increments)

## Implementation Plan

### New file: `falcon/core/local_logger.py`

```python
@ray.remote
class LocalLoggerWrapper:
    """Per-actor local logger, writes to graph/{actor_id}/metrics/"""

    def __init__(self, base_dir: str, actor_id: str, buffer_size: int = 100):
        self.base_dir = Path(base_dir) / actor_id / "metrics"
        self.buffer_size = buffer_size
        self.buffers = {}      # metric_name -> list of (step, value, walltime)
        self.counters = {}     # metric_name -> next step
        self.chunk_indices = {} # metric_name -> next chunk index

    def log(self, metrics: dict, step: Optional[int] = None):
        """Log metrics. Step auto-increments per metric if not provided."""
        walltime = time.time()
        for key, value in metrics.items():
            # Handle arrays/tensors
            if hasattr(value, 'numpy'):
                value = value.numpy()
            elif hasattr(value, 'item'):
                value = value.item()

            # Auto-increment step per metric
            if step is None:
                step_val = self.counters.get(key, 0)
                self.counters[key] = step_val + 1
            else:
                step_val = step

            # Buffer the entry
            if key not in self.buffers:
                self.buffers[key] = []
            self.buffers[key].append((step_val, value, walltime))

            # Flush if buffer full
            if len(self.buffers[key]) >= self.buffer_size:
                self._flush_metric(key)

    def _flush_metric(self, key: str):
        """Write buffer to NPZ chunk."""
        # ... write to chunk_{index}.npz

    def shutdown(self):
        """Flush all remaining buffers."""
        for key in self.buffers:
            if self.buffers[key]:
                self._flush_metric(key)
```

### Modify: `falcon/core/wandb_logger.py`

Add `LocalLoggerWrapper` alongside `WandBWrapper`:

```python
@ray.remote
class WandBManager:
    def __init__(self, ..., local_log_dir: Optional[str] = None):
        self.local_log_dir = local_log_dir
        self.local_loggers = {}  # actor_id -> LocalLoggerWrapper

    def init(self, actor_id: str, ...):
        # Existing wandb init
        self.wandb_runs[actor_id] = WandBWrapper.remote(...)
        # New local logger
        if self.local_log_dir:
            self.local_loggers[actor_id] = LocalLoggerWrapper.remote(
                self.local_log_dir, actor_id
            )

    def log(self, metrics, step, actor_id):
        # Existing wandb log
        self.wandb_runs[actor_id].log.remote(metrics, step=step)
        # New local log
        if actor_id in self.local_loggers:
            self.local_loggers[actor_id].log.remote(metrics, step=step)
```

### New file: `falcon/core/run_reader.py`

```python
class MetricReader:
    """Lazy-loaded metric from chunked NPZ files."""
    def __init__(self, metric_dir: Path):
        self.metric_dir = metric_dir
        self._values = None
        self._steps = None
        self._walltime = None

    @property
    def values(self) -> np.ndarray:
        if self._values is None:
            self._load()
        return self._values

    # ... similar for steps, walltime

class NodeReader:
    """Dict-like access to metrics for a node."""
    def __getitem__(self, metric_name: str) -> MetricReader:
        return MetricReader(self.node_dir / "metrics" / metric_name)

class RunReader:
    """Dict-like access to nodes in a run."""
    def __getitem__(self, node_name: str) -> NodeReader:
        return NodeReader(self.run_dir / node_name)

def read_run(path: str) -> RunReader:
    return RunReader(Path(path))
```

### Value types supported

- Scalars (int, float)
- numpy arrays
- torch tensors (converted via `.numpy()` or `.item()`)

## Design Decisions

1. **Array storage**: Same shape per metric, store as 2D array `(N_entries, *shape)`. Can extend later for variable shapes.
2. **Metric name sanitization**: Replace `:` with `_` for directory names. `Signal:mean` → `Signal_mean/`

# Non-goal

Functionality of the code should not change at all, only additional logging information should be generated.

