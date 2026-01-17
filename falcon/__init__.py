from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("falcon")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"

from .core.graph import Node, Graph, CompositeNode
from .core.deployed_graph import DeployedGraph
from .core.raystore import get_ray_dataset_manager
from .core.utils import LazyLoader
from .core.logger import Logger, get_logger, set_logger
from .core.run_reader import read_run
import falcon.contrib as contrib


def log(metrics: dict, step: int = None, prefix: str = None):
    """Log metrics using the current module-level logger.

    This is a convenience function for backward compatibility.
    Simulators and other code can use falcon.log({...}) to log metrics.

    If no logger is set (e.g., running outside of falcon context),
    this silently does nothing.

    Args:
        metrics: Dictionary of metric names to values
        step: Optional step number
        prefix: Optional prefix to add to all metric names
    """
    logger = get_logger()
    if logger is not None:
        logger.log(metrics, step=step, prefix=prefix)
