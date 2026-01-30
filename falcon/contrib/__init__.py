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
    "HypercubeMappingPrior": ".hypercubemappingprior",
    "TransformedPrior": ".product_prior",
    "ProductPrior": ".product_prior",
    "LazyOnlineNorm": ".norms",
    "Flow": ".flow",
    "NET_BUILDERS": ".flow",
    "build_mlp": ".networks",
    "EmbeddedPosterior": ".embedded_posterior",
    "StepwiseEstimator": ".stepwise_estimator",
    "LossBasedEstimator": ".stepwise_estimator",
    "TrainingLoopConfig": ".stepwise_estimator",
    "OptimizerConfig": ".stepwise_estimator",
    "InferenceConfig": ".stepwise_estimator",
    "SNPE_A": ".SNPE_A",
    "SNPEConfig": ".SNPE_A",
    "NetworkConfig": ".SNPE_A",
    "SNPEOptimizerConfig": ".SNPE_A",
    "SNPEInferenceConfig": ".SNPE_A",
    "SNPE_gaussian": ".SNPE_gaussian",
    "GaussianConfig": ".SNPE_gaussian",
    "GaussianPosteriorConfig": ".SNPE_gaussian",
    "GaussianPosterior": ".SNPE_gaussian",
}

# Map names that are aliased on import
_LAZY_ALIASES = {
    "SNPEOptimizerConfig": ("OptimizerConfig", ".SNPE_A"),
    "SNPEInferenceConfig": ("InferenceConfig", ".SNPE_A"),
}


def __getattr__(name):
    if name in _LAZY_ALIASES:
        real_name, module_path = _LAZY_ALIASES[name]
        import importlib
        module = importlib.import_module(module_path, __name__)
        return getattr(module, real_name)
    if name in _LAZY_IMPORTS:
        import importlib
        module = importlib.import_module(_LAZY_IMPORTS[name], __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
