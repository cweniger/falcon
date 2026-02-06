from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("falcon")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"

__all__ = [
    "Node", "Graph", "CompositeNode",
    "DeployedGraph",
    "get_ray_dataset_manager",
    "LazyLoader",
    "Logger", "get_logger", "set_logger", "log", "debug", "info", "warning", "error",
    "read_run",
    "load_run",
    "read_samples",
    "estimators",
    "priors",
    "embeddings",
]

_LAZY_IMPORTS = {
    "Node": ".core.graph",
    "Graph": ".core.graph",
    "CompositeNode": ".core.graph",
    "DeployedGraph": ".core.deployed_graph",
    "get_ray_dataset_manager": ".core.raystore",
    "LazyLoader": ".core.utils",
    "Logger": ".core.logger",
    "get_logger": ".core.logger",
    "set_logger": ".core.logger",
    "log": ".core.logger",
    "debug": ".core.logger",
    "info": ".core.logger",
    "warning": ".core.logger",
    "error": ".core.logger",
    "read_run": ".core.run_reader",
    "load_run": ".core.run_loader",
    "read_samples": ".core.samples_reader",
}


def __getattr__(name):
    if name in _LAZY_IMPORTS:
        import importlib
        module = importlib.import_module(_LAZY_IMPORTS[name], __name__)
        attr = getattr(module, name)
        globals()[name] = attr
        return attr
    if name in ("estimators", "priors", "embeddings"):
        import importlib
        mod = importlib.import_module(f".{name}", __name__)
        globals()[name] = mod
        return mod
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
