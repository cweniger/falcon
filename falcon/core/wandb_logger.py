"""
WandB logging backend.

This module provides WandB-specific logging implementations:
- WandBBackend: Non-Ray backend for single-process use
- WandBLoggerActor: Ray actor for distributed logging
- create_wandb_factory: Factory function for LoggerManager integration

WandB is an optional dependency. If not installed, attempting to use
WandB backends will raise an ImportError with installation instructions.
"""

import logging
from typing import Any, Dict, Optional

import ray

from .logger import LoggerBackend

# Optional wandb import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    wandb = None
    WANDB_AVAILABLE = False


def _check_wandb_available():
    """Raise ImportError if wandb is not installed."""
    if not WANDB_AVAILABLE:
        raise ImportError(
            "WandB is not installed. Install it with: pip install wandb"
        )


class WandBLogHandler(logging.Handler):
    """Forward log messages to WandB console output."""

    def __init__(self, run):
        super().__init__()
        self.run = run

    def emit(self, record):
        try:
            msg = self.format(record)
            # Log as console output to wandb (non-committing to avoid step issues)
            self.run.log({"_console": msg}, commit=False)
        except Exception:
            pass  # Don't crash on wandb errors


class WandBBackend(LoggerBackend):
    """WandB logging backend (non-Ray version).

    Logs metrics to Weights & Biases. Suitable for single-process use.

    Args:
        project: WandB project name.
        group: WandB group name.
        name: Run name.
        config: Run configuration dict.
        dir: Directory for WandB files.

    Raises:
        ImportError: If wandb is not installed.
    """

    def __init__(
        self,
        project: Optional[str] = None,
        group: Optional[str] = None,
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        dir: Optional[str] = None,
    ):
        _check_wandb_available()

        wandb_kwargs = {
            "project": project,
            "group": group,
            "name": name,
            "config": config or {},
            "reinit": True,
        }
        if dir:
            wandb_kwargs["dir"] = dir

        self.run = wandb.init(**wandb_kwargs)

        # Setup Python logging handler
        self._log_handler = WandBLogHandler(self.run)
        self._log_handler.setFormatter(logging.Formatter('%(message)s'))

    def log(
        self,
        metrics: Dict[str, Any],
        step: Optional[int] = None,
        walltime: Optional[float] = None,
    ) -> None:
        """Log metrics to WandB.

        Note: WandB manages its own timestamps; walltime parameter is accepted
        for interface compatibility but not used.
        """
        try:
            self.run.log(metrics, step=step)
        except Exception as e:
            print(f"WandB logging error: {e}")

    def get_log_handler(self) -> logging.Handler:
        """Return handler for Python logging integration."""
        return self._log_handler

    def shutdown(self) -> None:
        """Finish the WandB run."""
        self.run.finish()


@ray.remote
class WandBLoggerActor:
    """Ray actor wrapper for WandBBackend.

    Provides the same interface as WandBBackend but runs as a Ray actor
    for distributed logging scenarios.

    Raises:
        ImportError: If wandb is not installed.
    """

    def __init__(
        self,
        project: Optional[str] = None,
        group: Optional[str] = None,
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        dir: Optional[str] = None,
    ):
        self._backend = WandBBackend(
            project=project,
            group=group,
            name=name,
            config=config,
            dir=dir,
        )

    def log(
        self,
        metrics: Dict[str, Any],
        step: Optional[int] = None,
        walltime: Optional[float] = None,
    ) -> None:
        """Log metrics (delegates to WandBBackend)."""
        self._backend.log(metrics, step, walltime)

    def shutdown(self) -> None:
        """Shutdown the backend (delegates to WandBBackend)."""
        self._backend.shutdown()


def create_wandb_factory(
    project: Optional[str] = None,
    group: Optional[str] = None,
    dir: Optional[str] = None,
):
    """Create a WandB backend factory for use with LoggerManager.

    Args:
        project: WandB project name.
        group: WandB group name.
        dir: Directory for WandB files.

    Returns:
        Factory function that creates WandBLoggerActor instances.

    Raises:
        ImportError: If wandb is not installed.

    Example:
        manager = LoggerManager.remote({
            "wandb": create_wandb_factory(project="my_project"),
        })
    """
    _check_wandb_available()

    def factory(actor_id: str):
        return WandBLoggerActor.remote(
            project=project,
            group=group,
            name=actor_id,
            dir=dir,
        )

    return factory


# Backwards compatibility alias
WandBWrapper = WandBLoggerActor
