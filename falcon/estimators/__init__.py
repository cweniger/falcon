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

# Backward-compat aliases
SNPE_A = Flow
SNPEConfig = FlowConfig
SNPE_gaussian = Gaussian

# Re-export aliased config names from flow.py
from falcon.estimators.flow import (
    OptimizerConfig as SNPEOptimizerConfig,
    InferenceConfig as SNPEInferenceConfig,
)

__all__ = [
    # New names
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
    # Backward-compat aliases
    "SNPE_A",
    "SNPEConfig",
    "SNPEOptimizerConfig",
    "SNPEInferenceConfig",
    "SNPE_gaussian",
]
