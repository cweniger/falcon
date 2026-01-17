from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("falcon")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"

from .core.graph import Node, Graph, CompositeNode
from .core.deployed_graph import DeployedGraph
from .core.raystore import get_ray_dataset_manager
from .core.utils import LazyLoader
from .core.logger import Logger, get_logger, set_logger, log, debug, info, warning, error
from .core.run_reader import read_run
import falcon.contrib as contrib
