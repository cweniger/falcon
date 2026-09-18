"""Estimators for simulation-based inference.

Provides Flow (normalizing flow), FlowMatching (flow matching with a
truncated-prior proposal) and Gaussian posterior estimators, along with the
base class for torch-based estimators.

Flow requires the sbi package: pip install falcon-sbi[sbi]
"""

from falcon.estimators.torch_model import NetworkGroup, TorchModel
from falcon.estimators.gaussian_fullcov import GaussianFullCov
from falcon.estimators.flow_matching import FlowMatching

__all__ = [
    "Flow",
    "FlowMatching",
    "GaussianFullCov",
    "NetworkGroup",
    "TorchModel",
]

# Lazy imports for sbi-dependent classes
_LAZY_IMPORTS = {
    "Flow": "falcon.estimators.flow",
}


def __getattr__(name):
    if name in _LAZY_IMPORTS:
        import importlib
        module = importlib.import_module(_LAZY_IMPORTS[name])
        attr = getattr(module, name)
        globals()[name] = attr
        return attr
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
