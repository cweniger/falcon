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
# If no global logger is registered in the Ray cluster, logging is silently disabled.

import ray

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


def log(metrics: dict, step=None):
    if _logger_ref:
        _logger_ref.log.remote(metrics, step=step, actor_id=_actor_id)
