"""Estimators for simulation-based inference.

Provides Flow (normalizing flow) and Gaussian posterior estimators,
along with base classes for building custom estimators.

Flow requires the sbi package: pip install falcon-sbi[sbi]
"""

from falcon.estimators.stepwise_base import (
    StepwiseEstimator,
    LossBasedEstimator,
    TrainingLoopConfig,
)
from falcon.estimators.gaussian import (
    Gaussian,
    GaussianConfig,
    GaussianPosterior,
)
from falcon.estimators.gaussian_fullcov import GaussianFullCov

__all__ = [
    "Flow",
    "FlowConfig",
    "Gaussian",
    "GaussianConfig",
    "GaussianFullCov",
    "GaussianPosterior",
    "StepwiseEstimator",
    "LossBasedEstimator",
    "TrainingLoopConfig",
]

# Lazy imports for sbi-dependent classes
_LAZY_IMPORTS = {
    "Flow": "falcon.estimators.flow",
    "FlowConfig": "falcon.estimators.flow",
}


def __getattr__(name):
    if name in _LAZY_IMPORTS:
        import importlib
        module = importlib.import_module(_LAZY_IMPORTS[name])
        attr = getattr(module, name)
        globals()[name] = attr
        return attr
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
