"""Gaussian posterior estimation — Gaussian factory for LossBasedEstimator."""

from typing import List, Optional

from omegaconf import OmegaConf

from falcon.priors.product import TransformedPrior
from falcon.estimators.stepwise_base import LossBasedEstimator
from falcon.estimators.gaussian_fullcov import GaussianConfig, GaussianPosterior  # noqa: F401


# ==================== Factory Function ====================


def Gaussian(
    simulator_instance,
    theta_key: Optional[str] = None,
    condition_keys: Optional[List[str]] = None,
    config: Optional[dict] = None,
) -> LossBasedEstimator:
    """Create a LossBasedEstimator with GaussianPosterior.

    This is the main entry point for using Gaussian posterior estimation.
    It provides sensible defaults while allowing full customization.

    Args:
        simulator_instance: Prior/simulator instance
        theta_key: Key for theta in batch data
        condition_keys: Keys for condition data in batch
        config: Configuration dict with sections:
            - loop: TrainingLoopConfig options
            - network: NetworkConfig options
            - optimizer: OptimizerConfig options
            - inference: InferenceConfig options
            - embedding: Embedding configuration with _target_
            - device: Device string (optional)

    Returns:
        Configured LossBasedEstimator ready for training

    Example YAML:
        estimator:
          _target_: falcon.estimators.Gaussian
          network:
            hidden_dim: 128
            num_layers: 3
          embedding:
            _target_: model.E
            _input_: [x]
    """
    # Check simulator supports transformation interface
    if not isinstance(simulator_instance, TransformedPrior):
        raise TypeError(
            f"Gaussian requires a TransformedPrior (e.g., Product), "
            f"got {type(simulator_instance).__name__}. "
            f"The simulator must support forward/inverse with mode='standard_normal'."
        )

    # Merge with defaults
    schema = OmegaConf.structured(GaussianConfig)
    cfg = OmegaConf.merge(schema, config or {})

    # Extract configs as plain dicts
    embedding_config = OmegaConf.to_container(cfg.embedding, resolve=True)
    posterior_config = OmegaConf.to_container(cfg.network, resolve=True)

    return LossBasedEstimator(
        simulator_instance=simulator_instance,
        posterior_cls=GaussianPosterior,
        embedding_config=embedding_config,
        loop_config=cfg.loop,
        optimizer_config=cfg.optimizer,
        inference_config=cfg.inference,
        posterior_config=posterior_config,
        theta_key=theta_key,
        condition_keys=condition_keys,
        device=cfg.device,
        latent_mode="standard_normal",  # GaussianPosterior assumes N(0,I) prior
    )
