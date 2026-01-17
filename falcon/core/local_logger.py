"""
Local file-based logging backend.

This module provides a local logging backend that stores metrics in chunked NPZ files
and text messages in output.log files. It implements the LoggerBackend interface and
can be used standalone or as part of a CompositeLogger setup.

Storage structure:
    {base_dir}/{actor_id}/metrics/{metric_name}/chunk_{index}.npz
    {base_dir}/{actor_id}/output.log

Each NPZ file contains:
    - step: int64 array of step numbers
    - value: float64 array (scalar metrics) or 2D array (array metrics)
    - walltime: float64 array of epoch timestamps
"""

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import ray


class LoggerBackend:
    """Abstract base class for logging backends."""

    def log(
        self,
        metrics: Dict[str, Any],
        step: Optional[int] = None,
        walltime: Optional[float] = None,
    ) -> None:
        """Log a batch of metrics."""
        pass

    def shutdown(self) -> None:
        """Clean up resources."""
        pass

    def get_log_handler(self) -> Optional[logging.Handler]:
        """Return a logging.Handler for Python logging integration."""
        return None

# Log level names for text logging
_LEVEL_NAMES = {10: "DEBUG", 20: "INFO", 30: "WARNING", 40: "ERROR"}


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
        buffer_size: Number of entries to buffer before flushing to disk.
            Default is 100, balancing write frequency with data safety.
            Lower values (e.g., 10) provide more frequent saves but more I/O.
            Higher values (e.g., 1000) reduce I/O but risk data loss on crash.
    """

    def __init__(
        self,
        base_dir: str,
        name: str = "default",
        buffer_size: int = 100,
        text_buffer_size: int = 50,
        text_max_flush_interval: float = 5.0,
    ):
        self.base_dir = Path(base_dir)
        self.name = name
        self.metrics_dir = self.base_dir / name / "metrics"
        self.buffer_size = buffer_size

        # Per-metric state
        self.buffers: Dict[str, List[Tuple[int, Any, float]]] = {}
        self.counters: Dict[str, int] = {}
        self.chunk_indices: Dict[str, int] = {}

        # Text logging state
        self.text_buffer: List[str] = []
        self.text_buffer_size = text_buffer_size
        self.text_max_flush_interval = text_max_flush_interval
        self.last_text_flush = time.time()
        self.log_path = self.base_dir / name / "output.log"

        self.metrics_dir.mkdir(parents=True, exist_ok=True)

        # Create output.log immediately with startup message
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().isoformat(timespec="milliseconds")
        with open(self.log_path, "w") as f:
            f.write(f"{timestamp} [INFO] Logger initialized for '{name}'\n")

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

    def info(
        self,
        message: str,
        level: int = 20,
        walltime: Optional[float] = None,
    ) -> None:
        """Buffer text message, flush when buffer full or interval exceeded.

        Args:
            message: Text message to log
            level: Log level (DEBUG=10, INFO=20, WARNING=30, ERROR=40)
            walltime: Optional timestamp (epoch seconds)
        """
        if walltime is None:
            walltime = time.time()
        timestamp = datetime.fromtimestamp(walltime).isoformat(timespec="milliseconds")
        level_name = _LEVEL_NAMES.get(level, f"LVL{level}")
        line = f"{timestamp} [{level_name}] {message}\n"
        self.text_buffer.append(line)

        # Flush if buffer full or max interval exceeded
        now = time.time()
        if (
            len(self.text_buffer) >= self.text_buffer_size
            or now - self.last_text_flush >= self.text_max_flush_interval
        ):
            self._flush_text()

    def _flush_text(self) -> None:
        """Write buffered text messages to output.log."""
        if not self.text_buffer:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "a") as f:
            f.writelines(self.text_buffer)
        self.text_buffer = []
        self.last_text_flush = time.time()

    def get_log_handler(self) -> logging.Handler:
        """Return handler for Python logging integration."""
        return self._log_handler

    def get_log_tail(self, n: int = 50) -> List[str]:
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
        self._flush_text()
        self._log_handler.flush()

    def shutdown(self) -> None:
        """Flush all remaining buffers to disk."""
        self.flush()
        self._log_handler.close()


@ray.remote
class LocalLoggerActor:
    """Ray actor wrapper for LocalFileBackend.

    Provides the same interface as LocalFileBackend but runs as a Ray actor
    for distributed logging scenarios.

    Args:
        base_dir: Base directory for storing metrics.
        name: Identifier for this logger.
        buffer_size: Number of entries to buffer before flushing (default: 100).
        text_buffer_size: Max text messages before flush (default: 50).
        text_max_flush_interval: Max seconds before text flush (default: 5.0).
    """

    def __init__(
        self,
        base_dir: str,
        name: str = "default",
        buffer_size: int = 100,
        text_buffer_size: int = 50,
        text_max_flush_interval: float = 5.0,
    ):
        self._backend = LocalFileBackend(
            base_dir, name, buffer_size, text_buffer_size, text_max_flush_interval
        )

    def log(
        self,
        metrics: Dict[str, Any],
        step: Optional[int] = None,
        walltime: Optional[float] = None,
    ) -> None:
        """Log metrics (delegates to LocalFileBackend)."""
        self._backend.log(metrics, step, walltime)

    def info(
        self,
        message: str,
        level: int = 20,
        walltime: Optional[float] = None,
    ) -> None:
        """Log text message (delegates to LocalFileBackend)."""
        self._backend.info(message, level, walltime)

    def shutdown(self) -> None:
        """Shutdown the backend (delegates to LocalFileBackend)."""
        self._backend.shutdown()


def create_local_factory(
    base_dir: str,
    buffer_size: int = 100,
    text_buffer_size: int = 50,
    text_max_flush_interval: float = 5.0,
):
    """Create a local file backend factory for use with LoggerManager.

    Args:
        base_dir: Base directory for storing metrics.
        buffer_size: Number of entries to buffer before flushing (default: 100).
        text_buffer_size: Max text messages before flush (default: 50).
        text_max_flush_interval: Max seconds before text flush (default: 5.0).

    Returns:
        Factory function that creates LocalLoggerActor instances.

    Example:
        manager = LoggerManager.remote({
            "local": create_local_factory("/path/to/logs"),
        })
    """

    def factory(actor_id: str):
        return LocalLoggerActor.remote(
            base_dir=base_dir,
            name=actor_id,
            buffer_size=buffer_size,
            text_buffer_size=text_buffer_size,
            text_max_flush_interval=text_max_flush_interval,
        )

    return factory


# Backwards compatibility alias
LocalLoggerWrapper = LocalLoggerActor
