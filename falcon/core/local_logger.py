"""
Local file-based logging backend.

This module provides a local logging backend that stores metrics in chunked NPZ files.
It implements the LoggerBackend interface and can be used standalone or as part of
a CompositeLogger setup.

Storage structure:
    {base_dir}/{actor_id}/metrics/{metric_name}/chunk_{index}.npz

Each NPZ file contains:
    - step: int64 array of step numbers
    - value: float64 array (scalar metrics) or 2D array (array metrics)
    - walltime: float64 array of epoch timestamps
"""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import ray

from .logger import LoggerBackend


def sanitize_metric_name(name: str, actor_id: str = None) -> str:
    """Convert metric name to directory path.

    Transforms 'actor:group:metric' into 'group/metric' directory structure.
    The actor_id prefix is stripped since it's already part of the base path.

    Examples:
        'z:importance_sample:n_eff_min' -> 'importance_sample/n_eff_min'
        'train:loss' -> 'train/loss'
        'loss' -> 'loss'
    """
    # Strip actor_id prefix if present
    if actor_id and name.startswith(f"{actor_id}:"):
        name = name[len(actor_id) + 1:]

    # Use : as folder separator
    return name.replace(":", "/")


class LocalFileBackend(LoggerBackend):
    """Local file logging backend (non-Ray version).

    Writes metrics to chunked NPZ files on disk. Suitable for single-process
    use or when Ray is not needed.

    Args:
        base_dir: Base directory for storing metrics.
        name: Identifier for this logger (used in directory structure).
        buffer_size: Number of entries to buffer before flushing.
    """

    def __init__(self, base_dir: str, name: str = "default", buffer_size: int = 5):
        self.base_dir = Path(base_dir)
        self.name = name
        self.metrics_dir = self.base_dir / name / "metrics"
        self.buffer_size = buffer_size

        # Per-metric state
        self.buffers: Dict[str, List[Tuple[int, Any, float]]] = {}
        self.counters: Dict[str, int] = {}
        self.chunk_indices: Dict[str, int] = {}

        self.metrics_dir.mkdir(parents=True, exist_ok=True)

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """Log metrics to buffer, flushing when buffer is full."""
        walltime = time.time()

        for key, value in metrics.items():
            value = self._normalize_value(value)

            # Auto-increment step per metric if not provided
            if step is None:
                step_val = self.counters.get(key, 0)
                self.counters[key] = step_val + 1
            else:
                step_val = step
                self.counters[key] = max(self.counters.get(key, 0), step + 1)

            if key not in self.buffers:
                self.buffers[key] = []

            self.buffers[key].append((step_val, value, walltime))

            if len(self.buffers[key]) >= self.buffer_size:
                self._flush_metric(key)

    def _normalize_value(self, value: Any) -> Union[float, np.ndarray]:
        """Convert various value types to float or numpy array."""
        if hasattr(value, "numpy") and callable(value.numpy):
            value = value.numpy()

        if isinstance(value, np.ndarray):
            if value.ndim == 0:
                return float(value.item())
            return value
        elif isinstance(value, (int, float)):
            return float(value)
        elif hasattr(value, "item"):
            return float(value.item())
        return float(value)

    def _flush_metric(self, key: str) -> None:
        """Write buffered entries for a metric to a new NPZ chunk."""
        if not self.buffers.get(key):
            return

        buffer = self.buffers[key]
        sanitized_name = sanitize_metric_name(key, actor_id=self.name)
        metric_dir = self.metrics_dir / sanitized_name
        metric_dir.mkdir(parents=True, exist_ok=True)

        chunk_idx = self.chunk_indices.get(key, 0)
        chunk_path = metric_dir / f"chunk_{chunk_idx:06d}.npz"

        steps = np.array([entry[0] for entry in buffer], dtype=np.int64)
        walltime = np.array([entry[2] for entry in buffer], dtype=np.float64)

        first_value = buffer[0][1]
        if isinstance(first_value, np.ndarray):
            values = np.stack([entry[1] for entry in buffer])
        else:
            values = np.array([entry[1] for entry in buffer], dtype=np.float64)

        np.savez(chunk_path, step=steps, value=values, walltime=walltime)

        self.chunk_indices[key] = chunk_idx + 1
        self.buffers[key] = []

    def shutdown(self) -> None:
        """Flush all remaining buffers to disk."""
        for key in list(self.buffers.keys()):
            if self.buffers[key]:
                self._flush_metric(key)


@ray.remote
class LocalLoggerActor:
    """Ray actor wrapper for LocalFileBackend.

    Provides the same interface as LocalFileBackend but runs as a Ray actor
    for distributed logging scenarios.

    Args:
        base_dir: Base directory for storing metrics.
        name: Identifier for this logger.
        buffer_size: Number of entries to buffer before flushing.
    """

    def __init__(self, base_dir: str, name: str = "default", buffer_size: int = 5):
        self._backend = LocalFileBackend(base_dir, name, buffer_size)

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """Log metrics (delegates to LocalFileBackend)."""
        self._backend.log(metrics, step)

    def shutdown(self) -> None:
        """Shutdown the backend (delegates to LocalFileBackend)."""
        self._backend.shutdown()


def create_local_factory(base_dir: str, buffer_size: int = 5):
    """Create a local file backend factory for use with LoggerManager.

    Args:
        base_dir: Base directory for storing metrics.
        buffer_size: Number of entries to buffer before flushing.

    Returns:
        Factory function that creates LocalLoggerActor instances.

    Example:
        manager = LoggerManager.remote({
            "local": create_local_factory("/path/to/logs"),
        })
    """

    def factory(actor_id: str, config: Optional[Dict[str, Any]] = None):
        return LocalLoggerActor.remote(
            base_dir=base_dir,
            name=actor_id,
            buffer_size=buffer_size,
        )

    return factory


# Backwards compatibility alias
LocalLoggerWrapper = LocalLoggerActor
