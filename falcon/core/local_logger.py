"""
Local file-based logging backend.

Stores metrics in chunked NPZ files and text logs in output.log files.

Storage structure:
    {base_dir}/{name}/metrics/{metric_name}/chunk_{index}.npz
    {base_dir}/{name}/output.log

NPZ files contain: step (int64), value (float64 or 2D), walltime (float64).
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

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
    """Local file logging backend.

    Writes metrics to chunked NPZ files and text to output.log via Python logging.

    Args:
        base_dir: Base directory for logs (creates {base_dir}/{name}/ structure)
        name: Logger name, used as subdirectory
        buffer_size: Metrics buffered before flush (default 100)
    """

    def __init__(
        self,
        base_dir: str,
        name: str = "default",
        buffer_size: int = 100,
    ):
        self.base_dir = Path(base_dir)
        self.name = name
        self.metrics_dir = self.base_dir / name / "metrics"
        self.buffer_size = buffer_size
        self.log_path = self.base_dir / name / "output.log"

        # Per-metric state
        self.buffers: Dict[str, List[Tuple[int, Any, float]]] = {}
        self.counters: Dict[str, int] = {}
        self.chunk_indices: Dict[str, int] = {}

        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        # Setup Python logging handler for text logging
        self._log_handler = logging.FileHandler(self.log_path)
        self._log_handler.setFormatter(logging.Formatter(
            '%(asctime)s [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%dT%H:%M:%S'
        ))

    def log(
        self,
        metrics: Dict[str, Any],
        step: Optional[int] = None,
        walltime: Optional[float] = None,
    ) -> None:
        """Log metrics to buffer, flushing when buffer is full.

        Args:
            metrics: Dictionary mapping metric names to values.
            step: Optional step/iteration number.
            walltime: Optional timestamp (epoch seconds). If not provided,
                current time is used.
        """
        if walltime is None:
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

    def get_log_handler(self) -> logging.Handler:
        """Return handler for Python logging integration."""
        return self._log_handler

    def get_output_log_tail(self, n: int = 50) -> List[str]:
        """Read last n lines from disk."""
        if not self.log_path.exists():
            return []
        with open(self.log_path) as f:
            lines = f.readlines()
        return [line.rstrip() for line in lines[-n:]]

    def flush(self) -> None:
        """Flush all buffers to disk."""
        for key in list(self.buffers.keys()):
            if self.buffers[key]:
                self._flush_metric(key)
        self._log_handler.flush()

    def shutdown(self) -> None:
        """Flush all remaining buffers to disk."""
        self.flush()
        self._log_handler.close()
