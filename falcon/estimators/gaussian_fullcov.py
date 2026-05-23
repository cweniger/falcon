"""Full-covariance Gaussian estimator for TransformedPrior simulators."""

import copy
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from falcon.priors.product import TransformedPrior
from falcon.estimators.stepwise_base import StepwiseEstimator
from falcon.estimators.gaussian import GaussianConfig, GaussianPosterior
from falcon.core.logger import log, debug


class GaussianFullCov(StepwiseEstimator):
    """Full-covariance Gaussian posterior estimator for TransformedPrior simulators.

    Works in the standard-normal latent space defined by the simulator's
    forward/inverse transforms. Samples are mapped back to parameter space
    after generation.

    Gamma is resolved once at initialisation:
      - proposal sampling uses _proposal_gamma  (widens the distribution)
      - posterior sampling uses _posterior_gamma (corrects for proposal bias)
    """

    def __init__(
        self,
        simulator_instance,
        theta_key: Optional[str] = None,
        condition_keys: Optional[List[str]] = None,
        config: Optional[dict] = None,
    ):
        if not isinstance(simulator_instance, TransformedPrior):
            raise TypeError(
                f"GaussianFullCov requires a TransformedPrior (e.g., Product), "
                f"got {type(simulator_instance).__name__}."
            )

        schema = OmegaConf.structured(GaussianConfig)
        cfg = OmegaConf.merge(schema, config or {})

        super().__init__(
            simulator_instance=simulator_instance,
            loop_config=cfg.loop,
            theta_key=theta_key,
            condition_keys=condition_keys,
        )

        self.cfg = cfg

        if cfg.device:
            self.device = torch.device(cfg.device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            debug(f"Auto-detected device: {self.device}")

        # Resolve gamma once: proposal widens, posterior corrects back
        gamma = cfg.inference.gamma
        self._proposal_gamma = gamma
        self._posterior_gamma = (1.0 + gamma) / gamma if gamma is not None else None

        # Model state (initialised lazily on first batch)
        self._model: Optional[nn.Module] = None
        self._best_model: Optional[nn.Module] = None
        self._best_loss: float = float("inf")
        self._init_theta: Optional[torch.Tensor] = None
        self._init_conditions: Optional[Dict[str, torch.Tensor]] = None
        self._optimizer = None
        self._scheduler = None

    # ==================== Model Building ====================

    def _build_model(self, batch) -> nn.Module:
        from falcon.estimators.embedded_posterior import EmbeddedPosterior
        from falcon.embeddings import instantiate_embedding

        theta = self._to_tensor(batch[f"{self.theta_key}.value"])
        conditions = {
            k: self._to_tensor(batch[f"{k}.value"])
            for k in self.condition_keys if f"{k}.value" in batch
        }
        self._init_theta = theta
        self._init_conditions = conditions
        return self._create_model(theta, conditions)

    def _create_model(self, theta: torch.Tensor, conditions: Dict[str, torch.Tensor]) -> nn.Module:
        from falcon.estimators.embedded_posterior import EmbeddedPosterior
        from falcon.embeddings import instantiate_embedding

        theta_latent = self.simulator_instance.inverse(theta, mode="standard_normal")

        embedding_config = OmegaConf.to_container(self.cfg.embedding, resolve=True)
        embedding = instantiate_embedding(embedding_config).to(self.device)
        embedding.eval()
        with torch.no_grad():
            conditions_device = {k: v.to(self.device) for k, v in conditions.items()}
            embedded = embedding(conditions_device)

        network_config = OmegaConf.to_container(self.cfg.network, resolve=True)
        posterior = GaussianPosterior(
            param_dim=theta_latent.shape[1],
            condition_dim=embedded.shape[1],
            **network_config,
        ).to(self.device)

        debug(f"GaussianFullCov model built: param_dim={theta_latent.shape[1]}")
        return EmbeddedPosterior(embedding, posterior)

    def _initialize_model(self, batch) -> None:
        self._model = self._build_model(batch)
        self._best_model = copy.deepcopy(self._model)
        self._best_model.load_state_dict(
            {k: v.clone() for k, v in self._model.state_dict().items()}
        )

        opt_cfg = self.cfg.optimizer
        self._optimizer = AdamW(self._model.parameters(), lr=opt_cfg.lr, betas=opt_cfg.betas)
        self._scheduler = ReduceLROnPlateau(
            self._optimizer,
            mode="min",
            factor=opt_cfg.lr_decay_factor,
            patience=opt_cfg.scheduler_patience,
        )
        self.networks_initialized = True
        debug("GaussianFullCov initialised.")

    # ==================== Loss ====================

    def _compute_loss(self, batch):
        theta = self._to_tensor(batch[f"{self.theta_key}.value"], self.device)
        theta_logprob = self._to_tensor(batch[f"{self.theta_key}.log_prob"])
        conditions = {
            k: self._to_tensor(batch[f"{k}.value"], self.device)
            for k in self.condition_keys if f"{k}.value" in batch
        }

        theta_latent = self.simulator_instance.inverse(theta, mode="standard_normal")

        ts = time.time()
        self.history["train_ids"].extend((ts, id) for id in batch._ids.tolist())

        loss = self._model.loss(theta_latent, conditions)

        if self.cfg.inference.discard_samples:
            with torch.no_grad():
                self._model.eval()
                log_prob = self._model.log_prob(theta_latent, conditions).cpu()
            discard_mask = (log_prob - theta_logprob) < self.cfg.inference.log_ratio_threshold
            batch.discard(discard_mask)

        return loss, {"loss": loss.item()}

    # ==================== StepwiseEstimator abstract methods ====================

    def train_step(self, batch) -> Dict[str, float]:
        if not self.networks_initialized:
            self._initialize_model(batch)

        self._optimizer.zero_grad()
        self._model.train()
        loss, metrics = self._compute_loss(batch)
        loss.backward()
        self._optimizer.step()
        return metrics

    def val_step(self, batch) -> Dict[str, float]:
        with torch.no_grad():
            self._model.eval()
            _, metrics = self._compute_loss(batch)
        return metrics

    def on_epoch_end(self, epoch: int, val_metrics: Dict[str, float]) -> Optional[Dict[str, float]]:
        val_loss = val_metrics.get("loss", float("inf"))

        if val_loss < self._best_loss:
            self._best_loss = val_loss
            self._best_model.load_state_dict(
                {k: v.clone() for k, v in self._model.state_dict().items()}
            )
            log({"checkpoint": epoch})

        self._scheduler.step(val_loss)
        lr = self._optimizer.param_groups[0]["lr"]
        log({"lr": lr})

        extra = {"lr": lr}
        posterior = self._model.posterior
        if hasattr(posterior, "_output_std"):
            extra["theta_std"] = posterior._output_std.mean().item()
        if hasattr(posterior, "_residual_eigvals"):
            extra["eigvals_mean"] = posterior._residual_eigvals.mean().item()
        return extra

    # ==================== Sampling ====================

    def sample_prior(self, num_samples: int, conditions=None) -> dict:
        if conditions:
            raise ValueError("Conditions are not supported for sample_prior.")
        samples = self.simulator_instance.simulate_batch(num_samples)
        return {"value": samples, "log_prob": np.zeros(num_samples)}

    def _sample(self, num_samples: int, conditions, gamma) -> dict:
        if not self.networks_initialized:
            return self.sample_prior(num_samples)

        assert conditions, "Conditions must be provided for sampling."

        conditions_device = {
            k: self._to_tensor(v, self.device).expand(num_samples, *v.shape[1:])
            for k, v in conditions.items()
        }

        with torch.no_grad():
            self._best_model.eval()
            samples_latent = self._best_model.sample(conditions_device, gamma=gamma)
            log_prob = self._best_model.log_prob(samples_latent, conditions_device)
            samples = self.simulator_instance.forward(samples_latent, mode="standard_normal")

        return {"value": samples.cpu().numpy(), "log_prob": log_prob.cpu().numpy()}

    def sample_posterior(self, num_samples: int, conditions=None) -> dict:
        return self._sample(num_samples, conditions, gamma=self._posterior_gamma)

    def sample_proposal(self, num_samples: int, conditions=None) -> dict:
        if self._total_epochs_trained < self.loop_config.prior_epochs:
            return self.sample_prior(num_samples)
        result = self._sample(num_samples, conditions, gamma=self._proposal_gamma)
        log({
            "sample_proposal:mean": result["value"].mean(),
            "sample_proposal:std": result["value"].std(),
            "sample_proposal:logprob": result["log_prob"].mean(),
        })
        return result

    # ==================== Save / Load ====================

    def save(self, node_dir) -> None:
        node_dir = Path(node_dir)
        if not self.networks_initialized:
            raise RuntimeError("Cannot save: model not initialised.")

        torch.save(self._best_model.state_dict(), node_dir / "model.pth")
        torch.save(
            {"theta": self._init_theta, "conditions": self._init_conditions},
            node_dir / "init_tensors.pth",
        )
        torch.save(self._total_epochs_trained, node_dir / "total_epochs_trained.pth")

        torch.save(self.history["train_ids"], node_dir / "train_id_history.pth")
        torch.save(self.history["val_ids"], node_dir / "validation_id_history.pth")
        torch.save(self.history["epochs"], node_dir / "epochs.pth")
        torch.save(self.history["train_loss"], node_dir / "loss_train_posterior.pth")
        torch.save(self.history["val_loss"], node_dir / "loss_val_posterior.pth")
        torch.save(self.history["n_samples"], node_dir / "n_samples_total.pth")
        torch.save(self.history["elapsed_min"], node_dir / "elapsed_minutes.pth")

    def load(self, node_dir) -> None:
        node_dir = Path(node_dir)

        data = torch.load(node_dir / "init_tensors.pth")
        self._init_theta = data["theta"]
        self._init_conditions = data["conditions"]
        self._model = self._create_model(self._init_theta, self._init_conditions)
        self._best_model = copy.deepcopy(self._model)

        opt_cfg = self.cfg.optimizer
        self._optimizer = AdamW(self._model.parameters(), lr=opt_cfg.lr, betas=opt_cfg.betas)
        self._scheduler = ReduceLROnPlateau(
            self._optimizer,
            mode="min",
            factor=opt_cfg.lr_decay_factor,
            patience=opt_cfg.scheduler_patience,
        )
        self.networks_initialized = True

        tep = node_dir / "total_epochs_trained.pth"
        self._total_epochs_trained = torch.load(tep) if tep.exists() else 0

        self._best_model.load_state_dict(torch.load(node_dir / "model.pth"))
