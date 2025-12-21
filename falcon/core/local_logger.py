"""
Local file-based logging system for metrics.

This module provides a local logging backend that stores metrics in chunked NPZ files,
complementing the wandb logging system. Each metric is stored in its own directory
with chunked files for append-friendly writes.

Storage structure:
    {base_dir}/{actor_id}/metrics/{metric_name}/chunk_{index}.npz

Each NPZ file contains:
    - step: int64 array of step numbers
    - value: float64 array (scalar metrics) or 2D array (array metrics)
    - walltime: float64 array of epoch timestamps
"""

import time
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
import ray


def sanitize_metric_name(name: str) -> str:
    """Sanitize metric name for use as directory name.

    Replaces ':' with '_' to ensure filesystem compatibility.
    """
    return name.replace(":", "_")


@ray.remote
class LocalLoggerWrapper:
    """Per-actor local logger that writes metrics to chunked NPZ files.

    Each metric maintains its own buffer that is flushed to disk when full
    or when shutdown() is called.

    Args:
        base_dir: Base directory for storing metrics (e.g., graph/)
        actor_id: Identifier for this actor/node
        buffer_size: Number of entries to buffer before flushing (default: 100)
    """

    def __init__(self, base_dir: str, actor_id: str, buffer_size: int = 5):
        self.base_dir = Path(base_dir)
        self.actor_id = actor_id
        self.metrics_dir = self.base_dir / actor_id / "metrics"
        self.buffer_size = buffer_size

        # Per-metric state
        self.buffers: Dict[str, List[Tuple[int, Any, float]]] = {}  # metric -> [(step, value, walltime), ...]
        self.counters: Dict[str, int] = {}  # metric -> next step
        self.chunk_indices: Dict[str, int] = {}  # metric -> next chunk index

        # Create base metrics directory
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """Log metrics to buffer, flushing when buffer is full.

        Args:
            metrics: Dictionary of metric_name -> value
            step: Optional step number (auto-increments per metric if not provided)
        """
        walltime = time.time()

        for key, value in metrics.items():
            # Convert tensors/arrays to numpy
            if hasattr(value, 'numpy') and callable(value.numpy):
                # Torch tensor -> numpy
                value = value.numpy()

            # Now handle numpy arrays and scalars
            if isinstance(value, np.ndarray):
                if value.ndim == 0:
                    # 0-dim array -> scalar
                    value = float(value.item())
                # else: keep as array
            elif isinstance(value, (int, float)):
                value = float(value)
            elif hasattr(value, 'item'):
                # Other scalar-like objects
                value = float(value.item())

            # Auto-increment step per metric if not provided
            if step is None:
                step_val = self.counters.get(key, 0)
                self.counters[key] = step_val + 1
            else:
                step_val = step
                # Update counter to be at least step + 1
                self.counters[key] = max(self.counters.get(key, 0), step + 1)

            # Initialize buffer if needed
            if key not in self.buffers:
                self.buffers[key] = []

            # Buffer the entry
            self.buffers[key].append((step_val, value, walltime))

            # Flush if buffer full
            if len(self.buffers[key]) >= self.buffer_size:
                self._flush_metric(key)

    def _flush_metric(self, key: str):
        """Write buffered entries for a metric to a new NPZ chunk."""
        if not self.buffers.get(key):
            return

        buffer = self.buffers[key]
        sanitized_name = sanitize_metric_name(key)
        metric_dir = self.metrics_dir / sanitized_name
        metric_dir.mkdir(parents=True, exist_ok=True)

        # Get next chunk index
        chunk_idx = self.chunk_indices.get(key, 0)
        chunk_path = metric_dir / f"chunk_{chunk_idx:06d}.npz"

        # Extract arrays from buffer
        steps = np.array([entry[0] for entry in buffer], dtype=np.int64)
        walltime = np.array([entry[2] for entry in buffer], dtype=np.float64)

        # Handle values - check if scalar or array
        first_value = buffer[0][1]
        if isinstance(first_value, np.ndarray):
            # Array values: stack into 2D array (N_entries, *shape)
            values = np.stack([entry[1] for entry in buffer])
        else:
            # Scalar values
            values = np.array([entry[1] for entry in buffer], dtype=np.float64)

        # Save to NPZ
        np.savez(chunk_path, step=steps, value=values, walltime=walltime)

        # Update state
        self.chunk_indices[key] = chunk_idx + 1
        self.buffers[key] = []

    def shutdown(self):
        """Flush all remaining buffers to disk."""
        for key in list(self.buffers.keys()):
            if self.buffers[key]:
                self._flush_metric(key)
