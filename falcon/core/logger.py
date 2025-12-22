"""
Modular logging system with pluggable backends.

This module provides an abstract logging interface that supports multiple
backends (WandB, local files, TensorBoard, etc.) through a common API.

Usage:
    # Single metric (ergonomic)
    logger.log("train/loss", 0.5, step=10)

    # Batch metrics
    logger.log({"train/loss": 0.5, "train/acc": 0.9}, step=10)
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Union

import ray


class LoggerBackend(ABC):
    """Abstract base class for logging backends.

    Implement this interface to add new logging destinations
    (e.g., TensorBoard, MLflow, custom storage).
    """

    @abstractmethod
    def log(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """Log a batch of metrics.

        Args:
            metrics: Dictionary mapping metric names to values.
            step: Optional step/iteration number.
        """
        pass

    @abstractmethod
    def shutdown(self) -> None:
        """Clean up resources (flush buffers, close connections, etc.)."""
        pass


class CompositeLogger:
    """Logger that dispatches to multiple backends.

    Provides a unified API for logging to multiple destinations simultaneously.
    Supports both single-metric and batch-metric logging styles.

    Example:
        logger = CompositeLogger([wandb_backend, local_backend])
        logger.log("loss", 0.5, step=10)  # single metric
        logger.log({"loss": 0.5, "acc": 0.9}, step=10)  # batch
    """

    def __init__(self, backends: Optional[List[LoggerBackend]] = None):
        """Initialize with a list of backends.

        Args:
            backends: List of LoggerBackend instances. Can be empty initially
                      and backends added later via add_backend().
        """
        self.backends: List[LoggerBackend] = backends or []

    def add_backend(self, backend: LoggerBackend) -> None:
        """Add a logging backend."""
        self.backends.append(backend)

    def log(
        self,
        key_or_metrics: Union[str, Dict[str, Any]],
        value: Optional[Any] = None,
        *,
        step: Optional[int] = None,
    ) -> None:
        """Log metrics to all backends.

        Supports two calling conventions:

        1. Single metric (key-value style):
           logger.log("train/loss", 0.5, step=10)

        2. Batch metrics (dict style):
           logger.log({"train/loss": 0.5, "val/loss": 0.6}, step=10)

        Args:
            key_or_metrics: Either a metric name (str) or dict of metrics.
            value: Metric value (required if key_or_metrics is a string).
            step: Optional step/iteration number.

        Raises:
            ValueError: If key_or_metrics is a string but value is None.
        """
        # Normalize to dict format
        if isinstance(key_or_metrics, dict):
            metrics = key_or_metrics
        else:
            if value is None:
                raise ValueError(
                    f"value is required when logging a single metric. "
                    f"Got key={key_or_metrics!r}, value=None"
                )
            metrics = {key_or_metrics: value}

        # Dispatch to all backends
        for backend in self.backends:
            backend.log(metrics, step=step)

    def shutdown(self) -> None:
        """Shutdown all backends."""
        for backend in self.backends:
            backend.shutdown()


class NullBackend(LoggerBackend):
    """No-op backend for testing or disabling logging."""

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        pass

    def shutdown(self) -> None:
        pass


# Type for backend factory functions
BackendFactory = Callable[[str, Optional[Dict[str, Any]]], Any]


@ray.remote
class LoggerManager:
    """Central manager for distributed logging with multiple backends.

    Manages per-actor logger instances and dispatches log calls to all
    configured backends. Backend types are registered via factory functions,
    making it easy to add new backends without modifying this class.

    Args:
        backend_factories: Dict mapping backend names to factory functions.
            Each factory takes (actor_id, config) and returns a Ray actor.
    """

    def __init__(self, backend_factories: Optional[Dict[str, BackendFactory]] = None):
        self.backend_factories = backend_factories or {}
        self.actor_backends: Dict[str, Dict[str, Any]] = {}

    def register_backend(self, name: str, factory: BackendFactory) -> None:
        """Register a backend factory.

        Args:
            name: Backend identifier (e.g., "wandb", "local").
            factory: Callable that creates a backend actor.
        """
        self.backend_factories[name] = factory

    def init(self, actor_id: str, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize all registered backends for an actor.

        Args:
            actor_id: Unique identifier for the actor.
            config: Configuration dict passed to backend factories.
        """
        self.actor_backends[actor_id] = {}
        for name, factory in self.backend_factories.items():
            self.actor_backends[actor_id][name] = factory(actor_id, config)

    def log(
        self,
        key_or_metrics: Any,
        value: Optional[Any] = None,
        *,
        step: Optional[int] = None,
        actor_id: str = None,
    ) -> None:
        """Log metrics to all backends for an actor.

        Supports two calling conventions:

        1. Single metric:
           manager.log("train/loss", 0.5, step=10, actor_id="node_z")

        2. Batch metrics:
           manager.log({"loss": 0.5, "acc": 0.9}, step=10, actor_id="node_z")
        """
        # Normalize to dict format
        if isinstance(key_or_metrics, dict):
            metrics = key_or_metrics
        else:
            if value is None:
                raise ValueError(
                    f"value is required when logging a single metric. "
                    f"Got key={key_or_metrics!r}, value=None"
                )
            metrics = {key_or_metrics: value}

        # Dispatch to all backends for this actor
        if actor_id in self.actor_backends:
            for backend in self.actor_backends[actor_id].values():
                backend.log.remote(metrics, step=step)

    def shutdown(self) -> None:
        """Shutdown all backends for all actors."""
        for backends in self.actor_backends.values():
            for backend in backends.values():
                ray.get(backend.shutdown.remote())
