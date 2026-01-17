"""
Unified logging for falcon nodes and driver.

Usage:
    logger = Logger("node_z", config)
    logger.info("Training started")
    logger.log({"loss": 0.5})
    logger.shutdown()
"""

import logging
import sys
import traceback
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional


class LoggerBackend:
    """Abstract base class for logging backends.

    All logging backends (local file, WandB, etc.) should implement this interface.
    """

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


# Import backends after LoggerBackend is defined to avoid circular imports
from .local_logger import LocalFileBackend
from .wandb_logger import WandBBackend, WANDB_AVAILABLE


class _StreamCapture:
    """Capture a stream (stderr) and forward to logger."""

    def __init__(self, original, logger, level=logging.ERROR):
        self._original = original
        self._logger = logger
        self._level = level
        self._buffer = ""

    def write(self, text):
        # Write to original (so it still appears in terminal if driver)
        self._original.write(text)

        # Buffer and log complete lines
        self._buffer += text
        while '\n' in self._buffer:
            line, self._buffer = self._buffer.split('\n', 1)
            if line.strip():
                self._logger.log(self._level, line)

    def flush(self):
        self._original.flush()
        if self._buffer.strip():
            self._logger.log(self._level, self._buffer.strip())
            self._buffer = ""

    def isatty(self):
        return self._original.isatty()

    def __getattr__(self, name):
        return getattr(self._original, name)


class Logger:
    """
    Unified logger for a single node/driver.

    Text logging (with levels):
        logger.debug("...")   # Level 10
        logger.info("...")    # Level 20
        logger.warning("...")  # Level 30
        logger.error("...")   # Level 40

    Metric logging:
        logger.log({"loss": 0.5, "accuracy": 0.9})
        logger.log({"loss": 0.4}, step=10)

    Exception capture:
        - All uncaught exceptions -> output.log
        - All stderr output -> output.log
        - All warnings -> output.log

    Lifecycle:
        logger.flush()
        logger.shutdown()
    """

    LEVELS = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
    }

    def __init__(self, name: str, config: dict, capture_exceptions: bool = True):
        """
        Args:
            name: Identifier ("driver", "z", "theta", etc.)
            config: Run-level logging config from YAML
            capture_exceptions: If True, capture stderr and uncaught exceptions
        """
        self.name = name
        self._backends: List = []
        self._original_stderr = None
        self._original_excepthook = None

        # Local backend
        local_cfg = config.get("local", {})
        if local_cfg.get("enabled", True):
            base_dir = Path(local_cfg.get("dir", "."))
            self._local = LocalFileBackend(
                base_dir=str(base_dir),
                name=name,
                buffer_size=local_cfg.get("buffer_size", 100),
            )
            self._backends.append(self._local)
        else:
            self._local = None

        # WandB backend
        wandb_cfg = config.get("wandb", {})
        wandb_enabled = wandb_cfg.get("enabled", False)
        if wandb_cfg.get("driver_only", False) and name != "driver":
            wandb_enabled = False

        if wandb_enabled and WANDB_AVAILABLE:
            self._wandb = WandBBackend(
                project=wandb_cfg.get("project"),
                group=wandb_cfg.get("group"),  # shared across all nodes
                name=name,                      # node-specific
                dir=wandb_cfg.get("dir"),
            )
            self._backends.append(self._wandb)
        else:
            self._wandb = None

        # Python logger for text
        self._logger = logging.getLogger(f"falcon.{name}")
        self._logger.setLevel(logging.DEBUG)
        self._logger.handlers.clear()
        self._logger.propagate = False

        # Add handlers from backends
        for backend in self._backends:
            handler = backend.get_log_handler()
            if handler:
                level_str = config.get(
                    "local" if backend == self._local else "wandb", {}
                ).get("level", "DEBUG")
                handler.setLevel(self.LEVELS.get(level_str, logging.DEBUG))
                self._logger.addHandler(handler)

        # Console handler (for driver or if configured)
        if name == "driver" or config.get("console", {}).get("enabled", False):
            console = logging.StreamHandler()
            console.setFormatter(logging.Formatter(
                '%(asctime)s [%(levelname)s] %(message)s',
                datefmt='%Y-%m-%dT%H:%M:%S'
            ))
            console.setLevel(self.LEVELS.get(
                config.get("console", {}).get("level", "INFO"),
                logging.INFO
            ))
            self._logger.addHandler(console)

        # Capture exceptions, stderr, and warnings
        if capture_exceptions:
            self._setup_exception_capture()

    def _setup_exception_capture(self):
        """Capture stderr and uncaught exceptions to output.log."""
        # Capture stderr (includes tracebacks from raised exceptions)
        self._original_stderr = sys.stderr
        sys.stderr = _StreamCapture(self._original_stderr, self._logger, level=logging.ERROR)

        # Capture uncaught exceptions
        self._original_excepthook = sys.excepthook
        sys.excepthook = self._exception_handler

        # Capture warnings via logging
        logging.captureWarnings(True)
        warnings_logger = logging.getLogger('py.warnings')
        warnings_logger.handlers.clear()
        for handler in self._logger.handlers:
            warnings_logger.addHandler(handler)

    def _exception_handler(self, exc_type, exc_value, exc_tb):
        """Log uncaught exceptions before they crash the process."""
        tb_str = ''.join(traceback.format_exception(exc_type, exc_value, exc_tb))
        self._logger.error(f"Uncaught exception:\n{tb_str}")
        # Still call original hook
        if self._original_excepthook:
            self._original_excepthook(exc_type, exc_value, exc_tb)

    def _restore_exception_capture(self):
        """Restore original stderr and excepthook."""
        if self._original_stderr:
            sys.stderr = self._original_stderr
            self._original_stderr = None
        if self._original_excepthook:
            sys.excepthook = self._original_excepthook
            self._original_excepthook = None

    # ---- Text Logging ----

    def debug(self, message: str) -> None:
        self._logger.debug(message)

    def info(self, message: str) -> None:
        self._logger.info(message)

    def warning(self, message: str) -> None:
        self._logger.warning(message)

    def error(self, message: str) -> None:
        self._logger.error(message)

    # ---- Metric Logging ----

    def log(
        self,
        metrics: Dict[str, Any],
        step: Optional[int] = None,
        prefix: Optional[str] = None,
    ) -> None:
        """Log metrics to all backends."""
        if prefix:
            metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}

        for backend in self._backends:
            backend.log(metrics, step=step)

    # ---- Lifecycle ----

    def flush(self) -> None:
        """Force flush all buffered data."""
        for backend in self._backends:
            if hasattr(backend, "flush"):
                backend.flush()

    def shutdown(self) -> None:
        """Flush and close all backends, restore stderr/excepthook."""
        self.flush()
        self._restore_exception_capture()
        for backend in self._backends:
            backend.shutdown()
        self._logger.handlers.clear()

    # ---- For Monitor ----

    def get_output_log_tail(self, n: int = 50) -> List[str]:
        """Get last n log lines (reads from disk)."""
        if self._local:
            return self._local.get_output_log_tail(n)
        return []


# ============================================================================
# Module-level logger and convenience functions
# ============================================================================


class _DefaultLogger:
    """Fallback logger that prints to console when no Logger is set.

    Useful for testing/debugging code outside of falcon context.
    Text messages print to console, metrics are silently ignored.
    """

    def debug(self, message: str) -> None:
        pass  # Silent for debug

    def info(self, message: str) -> None:
        print(f"[INFO] {message}")

    def warning(self, message: str) -> None:
        print(f"[WARNING] {message}")

    def error(self, message: str) -> None:
        print(f"[ERROR] {message}", file=sys.stderr)

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None,
            prefix: Optional[str] = None) -> None:
        pass  # Silent for metrics


_current_logger: Any = _DefaultLogger()


def get_logger() -> Any:
    """Get the current module-level logger instance."""
    return _current_logger


def set_logger(logger: Logger) -> None:
    """Set the current module-level logger instance."""
    global _current_logger
    _current_logger = logger


# ---- Module-level convenience functions ----
# These dispatch to _current_logger, allowing simulators and estimators
# to log without needing explicit logger parameters.

def log(metrics: Dict[str, Any], step: Optional[int] = None, prefix: Optional[str] = None) -> None:
    """Log metrics to the current logger.

    Args:
        metrics: Dictionary of metric names to values
        step: Optional step number
        prefix: Optional prefix to add to all metric names
    """
    _current_logger.log(metrics, step=step, prefix=prefix)


def debug(message: str) -> None:
    """Log a debug message."""
    _current_logger.debug(message)


def info(message: str) -> None:
    """Log an info message."""
    _current_logger.info(message)


def warning(message: str) -> None:
    """Log a warning message."""
    _current_logger.warning(message)


def error(message: str) -> None:
    """Log an error message."""
    _current_logger.error(message)
