# Note:
# This module provides a simple interface for logging metrics from Ray actors or workers
# to a central `LoggingActor`. It supports distributed simulation or training setups
# where multiple processes (actors) log to a shared target (e.g., wandb, files).
#
# - `initialize_logging_for(actor_id)` must be called once per process to register
#   the actor and obtain a reference to the global logger. It stores `actor_id` and
#   fetches the named actor `'falcon:global_logger'` from the Ray cluster.
#
# - `log(metrics, step)` can then be called repeatedly to asynchronously send metrics
#   to the logger. These are associated with the calling actor via `actor_id`.
#
# - `info(message, level)` can be called to log text messages with timestamps.
#
# If no global logger is registered in the Ray cluster, logging is silently disabled.

import sys

import ray

# Log level constants (matching Python's logging module)
DEBUG = 10
INFO = 20
WARNING = 30
ERROR = 40

_LEVEL_NAMES = {DEBUG: "DEBUG", INFO: "INFO", WARNING: "WARNING", ERROR: "ERROR"}

_logger_ref = None
_actor_id = None
_original_stdout = None
_original_stderr = None
_echo_to_stdout = False  # If True, info() also prints to terminal (for driver)
_falcon_log_file = None  # If set, info() also writes to falcon.log


def set_falcon_log(log_file):
    """Set the falcon.log file handle for info() to write to."""
    global _falcon_log_file
    _falcon_log_file = log_file


class _OutputCapture:
    """Captures stdout/stderr and forwards to info() logging."""

    def __init__(self, original, level, keep_original=False):
        self._original = original
        self._level = level
        self._keep_original = keep_original
        self._buffer = ""

    def write(self, text):
        # Optionally write to original (for driver to keep terminal output)
        if self._keep_original:
            self._original.write(text)

        # Buffer text and flush on newlines
        self._buffer += text
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            if line.strip():  # Skip empty lines
                info(line, self._level, _from_capture=True)

    def flush(self):
        # Flush any remaining buffered content
        if self._buffer.strip():
            info(self._buffer.strip(), self._level, _from_capture=True)
            self._buffer = ""
        self._original.flush()

    def isatty(self):
        return self._original.isatty() if self._keep_original else False

    # Forward other attributes to original
    def __getattr__(self, name):
        return getattr(self._original, name)


def initialize_logging_for(actor_id, capture_output=True, keep_original=False):
    """
    Called by the orchestrating code (Ray worker), once per process or actor.
    Registers a remote LoggingActor handle and the actor_id.

    Args:
        actor_id: Identifier for this actor/process
        capture_output: If True, redirect stdout/stderr to output.log
        keep_original: If True, also write to original stdout/stderr (for driver)
    """
    global _logger_ref, _actor_id, _original_stdout, _original_stderr, _echo_to_stdout
    _actor_id = actor_id
    _echo_to_stdout = keep_original  # Echo info() to terminal on driver
    try:
        _logger_ref = ray.get_actor(name="falcon:global_logger")
    except ValueError:
        print("Global logger actor not found. Logging information will be not saved.")
    if _logger_ref:
        _logger_ref.init.remote(actor_id=actor_id)

        # Capture stdout/stderr and redirect to output.log
        if capture_output:
            _original_stdout = sys.stdout
            _original_stderr = sys.stderr
            sys.stdout = _OutputCapture(_original_stdout, INFO, keep_original)
            sys.stderr = _OutputCapture(_original_stderr, WARNING, keep_original)


def log(metrics: dict, log_prefix=None):
    if _logger_ref:
        metrics = {
            (f"{_actor_id}:" if _actor_id else "") + (f"{log_prefix}:" if log_prefix else "") + k: v for k, v in metrics.items()
        }
        _logger_ref.log.remote(metrics, step=None, actor_id=_actor_id)


def info(message: str, level: int = INFO, _from_capture: bool = False):
    """Log a text message with timestamp.

    Args:
        message: Text message to log
        level: Log level (DEBUG=10, INFO=20, WARNING=30, ERROR=40)
        _from_capture: Internal flag - True when called from _OutputCapture
    """
    from datetime import datetime
    timestamp = datetime.now().isoformat(timespec="milliseconds")
    level_name = _LEVEL_NAMES.get(level, "INFO")
    formatted_line = f"{timestamp} [{level_name}] {message}\n"

    # Echo to terminal on driver (for user visibility)
    # Skip if called from _OutputCapture (it already wrote to original stdout)
    if _echo_to_stdout and _original_stdout and not _from_capture:
        _original_stdout.write(formatted_line)
        _original_stdout.flush()

    # Write to falcon.log if set
    if _falcon_log_file and not _from_capture:
        _falcon_log_file.write(formatted_line)
        _falcon_log_file.flush()

    if _logger_ref:
        _logger_ref.info.remote(message, level=level, actor_id=_actor_id)
