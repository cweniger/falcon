"""Estimators for simulation-based inference.

Provides Flow (normalizing flow) and Gaussian posterior estimators,
along with base classes for building custom estimators.
"""

from falcon.estimators.base import (
    StepwiseEstimator,
    LossBasedEstimator,
    TrainingLoopConfig,
    OptimizerConfig,
    InferenceConfig,
)
from falcon.estimators.flow import (
    Flow,
    FlowConfig,
    NetworkConfig,
)
from falcon.estimators.gaussian import (
    Gaussian,
    GaussianConfig,
    GaussianPosteriorConfig,
    GaussianPosterior,
)

__all__ = [
    "Flow",
    "FlowConfig",
    "NetworkConfig",
    "Gaussian",
    "GaussianConfig",
    "GaussianPosteriorConfig",
    "GaussianPosterior",
    "StepwiseEstimator",
    "LossBasedEstimator",
    "TrainingLoopConfig",
    "OptimizerConfig",
    "InferenceConfig",
]
