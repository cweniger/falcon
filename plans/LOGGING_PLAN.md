# Plan: Add Text Logging to Replace print() Statements

## Current State

**Metrics logging (`log()`):**
- `log({"key": value})` logs metrics (numeric values)
- Stored in chunked NPZ files: `{base_dir}/{actor_id}/metrics/{metric_name}/chunk_*.npz`
- Supports distributed logging via Ray actors

**Print statements (~42 total):**
- STATUS: "Training started...", "Graph ready.", "Epoch 1/2"
- INFO: "Saved config to...", "Observation shapes..."
- DEBUG: "Auto-detected device...", "Condition keys..."
- WARNING: "wandb not installed..."
- HELP: CLI usage text

## Proposed Design

### New Function: `info(message)`

Add a text logging function parallel to `log()`:

```python
# falcon/core/logging.py

def info(message: str, level: str = "info"):
    """Log a text message with timestamp.

    Args:
        message: Text message to log
        level: One of "debug", "info", "warning", "error"
    """
```

### Storage Format

Text logs stored as line-delimited file:
```
{base_dir}/{actor_id}/console.log
```

Format per line:
```
2024-01-09T18:30:45.123 [INFO] Training started for: z
2024-01-09T18:30:46.456 [INFO] Epoch 1/2
```

### Implementation Changes

**1. Add `info()` function to `falcon/core/logging.py`:**
```python
def info(message: str, level: str = "info"):
    """Log a text message."""
    if _logger_ref:
        _logger_ref.info.remote(message, level=level, actor_id=_actor_id)
```

**2. Add `info()` method to `LoggerManager` in `falcon/core/logger.py`:**
```python
def info(self, message: str, level: str = "info", actor_id: str = None):
    """Log text message to all backends."""
    walltime = time.time()
    if actor_id in self.actor_backends:
        for backend in self.actor_backends[actor_id].values():
            backend.info.remote(message, level=level, walltime=walltime)
```

**3. Add `info()` method to `LocalLoggerActor` in `falcon/core/local_logger.py`:**
```python
def info(self, message: str, level: str = "info", walltime: float = None):
    """Append text message to console.log."""
    if walltime is None:
        walltime = time.time()
    timestamp = datetime.fromtimestamp(walltime).isoformat(timespec='milliseconds')
    log_path = self.base_dir / self.name / "console.log"
    with open(log_path, "a") as f:
        f.write(f"{timestamp} [{level.upper()}] {message}\n")
```

**4. Replace print() statements:**

| Location | Current | Replacement |
|----------|---------|-------------|
| `deployed_graph.py:200` | `print("Spinning up graph...")` | `info("Spinning up graph...")` |
| `deployed_graph.py:106` | `print(f"Training started for: {name}")` | `info(f"Training started for: {name}")` |
| `SNPE_A.py:145` | `print(f"Auto-detected device: {device}")` | `info(f"Auto-detected device: {device}", level="debug")` |
| etc. | | |

**5. For CLI output (non-Ray context):**

Some print statements in `cli.py` happen before Ray is initialized (graph visualization, help text). These should remain as `print()` since they're user-facing CLI output, not distributed logs.

Keep as `print()`:
- Help/usage text
- Graph structure visualization (table)
- "Saved config to..." (happens before Ray)
- Sample generation progress

Convert to `info()`:
- Messages inside Ray actors (deployed_graph.py, SNPE_A.py, stepwise_estimator.py)

### File Changes Summary

1. **`falcon/core/logging.py`** - Add `info()` function
2. **`falcon/core/logger.py`** - Add `info()` to LoggerManager
3. **`falcon/core/local_logger.py`** - Add `info()` to LocalLoggerActor and LocalFileBackend
4. **`falcon/core/deployed_graph.py`** - Replace ~7 print statements
5. **`falcon/contrib/SNPE_A.py`** - Replace ~6 print statements
6. **`falcon/contrib/stepwise_estimator.py`** - Replace ~1 print statement

### Questions for User

1. **Log file name:** `console.log` vs `messages.log` vs `output.log`?
2. **Should CLI output also go to log file?** Or keep print() for immediate user feedback?
3. **Buffer/flush strategy:** Flush after each message (immediate), or buffer like metrics?
