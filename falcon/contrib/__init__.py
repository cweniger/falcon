"""Backward-compatibility shim.

All classes have moved to falcon.estimators, falcon.priors, and falcon.embeddings.
This module re-exports them for backward compatibility.
"""

__all__ = [
    "HypercubeMappingPrior",
    "TransformedPrior", "ProductPrior",
    "LazyOnlineNorm",
    "Flow", "NET_BUILDERS",
    "build_mlp",
    "EmbeddedPosterior",
    "StepwiseEstimator", "LossBasedEstimator", "TrainingLoopConfig", "OptimizerConfig", "InferenceConfig",
    "SNPE_A", "SNPEConfig", "NetworkConfig", "SNPEOptimizerConfig", "SNPEInferenceConfig",
    "SNPE_gaussian", "GaussianConfig", "GaussianPosteriorConfig", "GaussianPosterior",
]

_LAZY_IMPORTS = {
    # Priors
    "HypercubeMappingPrior": "falcon.priors",
    "TransformedPrior": "falcon.priors",
    "ProductPrior": "falcon.priors",
    # Embeddings
    "LazyOnlineNorm": "falcon.embeddings",
    "DiagonalWhitener": "falcon.embeddings",
    "hartley_transform": "falcon.embeddings",
    "PCAProjector": "falcon.embeddings",
    "instantiate_embedding": "falcon.embeddings",
    # Estimator internals
    "Flow": "falcon.estimators.flow_density",
    "NET_BUILDERS": "falcon.estimators.flow_density",
    "build_mlp": "falcon.estimators.networks",
    "EmbeddedPosterior": "falcon.estimators.embedded_posterior",
    # Base estimators
    "StepwiseEstimator": "falcon.estimators.base",
    "LossBasedEstimator": "falcon.estimators.base",
    "TrainingLoopConfig": "falcon.estimators.base",
    # Estimators
    "SNPE_A": "falcon.estimators",
    "SNPEConfig": "falcon.estimators",
    "NetworkConfig": "falcon.estimators",
    "SNPE_gaussian": "falcon.estimators",
    "GaussianConfig": "falcon.estimators",
    "GaussianPosteriorConfig": "falcon.estimators",
    "GaussianPosterior": "falcon.estimators",
}

# Aliased names: the attribute name in this module differs from the source name
_LAZY_ALIASES = {
    "SNPEOptimizerConfig": ("OptimizerConfig", "falcon.estimators.flow"),
    "SNPEInferenceConfig": ("InferenceConfig", "falcon.estimators.flow"),
    "OptimizerConfig": ("OptimizerConfig", "falcon.estimators.base"),
    "InferenceConfig": ("InferenceConfig", "falcon.estimators.base"),
}


def __getattr__(name):
    if name in _LAZY_ALIASES:
        real_name, module_path = _LAZY_ALIASES[name]
        import importlib
        module = importlib.import_module(module_path)
        return getattr(module, real_name)
    if name in _LAZY_IMPORTS:
        import importlib
        module = importlib.import_module(_LAZY_IMPORTS[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
