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

import ray

# Log level constants (matching Python's logging module)
DEBUG = 10
INFO = 20
WARNING = 30
ERROR = 40

_LEVEL_NAMES = {DEBUG: "DEBUG", INFO: "INFO", WARNING: "WARNING", ERROR: "ERROR"}

_logger_ref = None
_actor_id = None


def initialize_logging_for(actor_id):
    """
    Called by the orchestrating code (Ray worker), once per process or actor.
    Registers a remote LoggingActor handle and the actor_id.
    """
    global _logger_ref, _actor_id
    _actor_id = actor_id
    try:
        _logger_ref = ray.get_actor(name="falcon:global_logger")
    except ValueError:
        print("Global logger actor not found. Logging information will be not saved.")
    if _logger_ref:
        _logger_ref.init.remote(actor_id=actor_id)


def log(metrics: dict, log_prefix=None):
    if _logger_ref:
        metrics = {
            (f"{_actor_id}:" if _actor_id else "") + (f"{log_prefix}:" if log_prefix else "") + k: v for k, v in metrics.items()
        }
        _logger_ref.log.remote(metrics, step=None, actor_id=_actor_id)


def info(message: str, level: int = INFO):
    """Log a text message with timestamp.

    Args:
        message: Text message to log
        level: Log level (DEBUG=10, INFO=20, WARNING=30, ERROR=40)
    """
    if _logger_ref:
        _logger_ref.info.remote(message, level=level, actor_id=_actor_id)
