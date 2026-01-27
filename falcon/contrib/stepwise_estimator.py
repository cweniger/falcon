"""Stepwise estimator with epoch-based training loop."""

import asyncio
import copy
import time
from abc import abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from falcon.core.base_estimator import BaseEstimator
from falcon.core.utils import RVBatch
from falcon.core.logger import log, debug, info, warning, error


@dataclass
class TrainingLoopConfig:
    """Generic training loop parameters."""

    num_epochs: int = 100
    batch_size: int = 128
    early_stop_patience: int = 16
    reset_network_after_pause: bool = False


@dataclass
class OptimizerConfig:
    """Optimizer and scheduler parameters."""

    lr: float = 1e-3
    betas: tuple = (0.9, 0.9)  # Lower beta2 for dynamic SBI setting
    lr_decay_factor: float = 0.1
    scheduler_patience: int = 8


@dataclass
class InferenceConfig:
    """Inference and sampling parameters."""

    gamma: float = 0.5
    discard_samples: bool = False
    log_ratio_threshold: float = -20.0


class StepwiseEstimator(BaseEstimator):
    """
    Estimator with epoch-based training loop.

    Provides concrete implementations for:
    - train() with epoch iteration and early stopping
    - pause/resume/interrupt

    Subclasses must implement:
    - train_step() / val_step() / on_epoch_end()
    - sample_prior/posterior/proposal
    - save/load
    """

    def __init__(
        self,
        simulator_instance,
        loop_config: TrainingLoopConfig,
        theta_key: Optional[str] = None,
        condition_keys: Optional[List[str]] = None,
    ):
        """
        Initialize the stepwise estimator.

        Args:
            simulator_instance: Prior/simulator instance
            loop_config: Training loop configuration
            theta_key: Key for theta in batch data
            condition_keys: Keys for condition data in batch
        """
        self.simulator_instance = simulator_instance
        self.loop_config = loop_config
        self.param_dim = simulator_instance.param_dim

        # Key configuration for Batch access
        self.theta_key = theta_key
        self.condition_keys = condition_keys or []

        # Async control
        self._pause_event = asyncio.Event()
        self._pause_event.set()
        self._terminated = False
        self._break_flag = False

        # Networks initialized flag (managed by subclass)
        self.networks_initialized = False

        # History tracking
        self.history = {
            "train_ids": [],
            "val_ids": [],
            "epochs": [],
            "train_loss": [],
            "val_loss": [],
            "n_samples": [],
            "elapsed_min": [],
        }

    # ==================== Abstract Methods ====================

    @abstractmethod
    def train_step(self, batch) -> Dict[str, float]:
        """
        Execute one training step.

        Args:
            batch: Batch object containing numpy arrays accessible via batch[key]

        Returns:
            Dict of metrics to log. Must include "loss" key.

        Side effects:
            - Move data to GPU as appropriate
            - Apply implementation-specific transforms
            - Call batch.discard(mask) to mark irrelevant samples
            - Update optimizer
        """
        pass

    @abstractmethod
    def val_step(self, batch) -> Dict[str, float]:
        """
        Execute one validation step.

        Args:
            batch: Batch object containing numpy arrays

        Returns:
            Dict of metrics to log. Must include "loss" key for early stopping.

        Note:
            NO batch.discard() calls - validation does not affect sample lifecycle.
        """
        pass

    @abstractmethod
    def on_epoch_end(self, epoch: int, val_metrics: Dict[str, float]) -> None:
        """
        Hook called at end of each epoch.

        Use for:
        - Updating best model weights
        - LR scheduler step
        - Custom checkpointing logic

        Args:
            epoch: Current epoch number (0-indexed)
            val_metrics: Validation metrics from this epoch
        """
        pass

    # ==================== Concrete Methods ====================

    async def train(self, buffer) -> None:
        """
        Main training loop with epochs and early stopping.

        Args:
            buffer: BufferView providing access to training/validation data
        """
        cfg = self.loop_config

        # Setup dataloaders
        keys = [self.theta_key, f"{self.theta_key}.logprob", *self.condition_keys]
        dataloader_train = buffer.train_loader(keys, batch_size=cfg.batch_size)
        dataloader_val = buffer.val_loader(keys, batch_size=cfg.batch_size)

        best_val_loss = float("inf")
        epochs_no_improve = 0
        t0 = time.perf_counter()

        for epoch in range(cfg.num_epochs):
            info(f"Epoch {epoch+1}/{cfg.num_epochs}")
            log({"epoch": epoch + 1})

            # === Training phase ===
            train_metrics_sum = {}
            num_train_batches = 0

            for batch in dataloader_train:
                metrics = self.train_step(batch)

                # Accumulate metrics
                for k, v in metrics.items():
                    train_metrics_sum[k] = train_metrics_sum.get(k, 0) + v
                num_train_batches += 1

                # Log step-level metrics
                for k, v in metrics.items():
                    log({f"train:{k}": v})

                # Async yield and pause check
                await asyncio.sleep(0)
                await self._pause_event.wait()
                if self._break_flag:
                    self._break_flag = False
                    break

            # Average training metrics
            train_metrics_avg = {
                k: v / num_train_batches for k, v in train_metrics_sum.items()
            }

            # === Validation phase ===
            val_metrics_sum = {}
            num_val_samples = 0

            for batch in dataloader_val:
                metrics = self.val_step(batch)
                batch_size = len(batch)

                # Accumulate (sum for later averaging)
                for k, v in metrics.items():
                    val_metrics_sum[k] = val_metrics_sum.get(k, 0) + v * batch_size
                num_val_samples += batch_size

                await asyncio.sleep(0)
                await self._pause_event.wait()
                if self._break_flag:
                    self._break_flag = False
                    break

            # Average validation metrics
            val_metrics_avg = {
                k: v / num_val_samples for k, v in val_metrics_sum.items()
            }

            # Log validation metrics
            for k, v in val_metrics_avg.items():
                log({f"val:{k}": v})

            # === End of epoch hook ===
            self.on_epoch_end(epoch, val_metrics_avg)

            # === Early stopping ===
            val_loss = val_metrics_avg.get("loss", float("inf"))
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            # Record history
            self._record_epoch_history(
                epoch, train_metrics_avg, val_metrics_avg, t0, buffer
            )

            if epochs_no_improve >= cfg.early_stop_patience:
                info("Early stopping triggered.")
                break

            await self._pause_event.wait()
            if self._terminated:
                break

    def _record_epoch_history(
        self, epoch, train_metrics, val_metrics, t0, buffer
    ) -> None:
        """Record metrics to history dict."""
        self.history["epochs"].append(epoch + 1)
        self.history["train_loss"].append(train_metrics.get("loss", 0))
        self.history["val_loss"].append(val_metrics.get("loss", 0))

        elapsed = (time.perf_counter() - t0) / 60.0
        self.history["elapsed_min"].append(elapsed)
        log({"elapsed_minutes": elapsed})

        try:
            stats = buffer.get_stats()
            self.history["n_samples"].append(stats["total_length"])
        except Exception:
            pass

    # ==================== Pause/Resume/Interrupt ====================

    def pause(self) -> None:
        """Pause training loop."""
        self._pause_event.clear()

    def resume(self) -> None:
        """Resume training loop."""
        if self.loop_config.reset_network_after_pause:
            self.networks_initialized = False
            self._break_flag = True
        self._pause_event.set()

    def interrupt(self) -> None:
        """Terminate training loop."""
        self._terminated = True
        self._pause_event.set()


# ==================== LossBasedEstimator ====================


class LossBasedEstimator(StepwiseEstimator):
    """Complete estimator that trains a posterior model by minimizing loss.

    This class provides a full implementation with:
    - Model creation from posterior_cls + embedding
    - Batch extraction and loss computation
    - Sampling (prior, posterior, proposal)
    - Save/load with init tensor storage

    The posterior must implement the Posterior contract:
        - loss(theta, conditions) -> Tensor
        - sample(conditions, gamma=None) -> Tensor
        - log_prob(theta, conditions) -> Tensor
    """

    def __init__(
        self,
        simulator_instance,
        posterior_cls: Type[nn.Module],
        embedding_config: dict,
        loop_config: TrainingLoopConfig,
        optimizer_config: OptimizerConfig,
        inference_config: InferenceConfig,
        posterior_config: Optional[dict] = None,
        theta_key: Optional[str] = None,
        condition_keys: Optional[List[str]] = None,
        device: Optional[str] = None,
        latent_mode: Optional[str] = None,
    ):
        """
        Args:
            latent_mode: Transform theta to latent space for training/sampling.
                None: work directly in theta space (default)
                "standard_normal": transform to N(0,I) via simulator.inverse/forward
                "hypercube": transform to hypercube via simulator.inverse/forward
        """
        super().__init__(
            simulator_instance=simulator_instance,
            loop_config=loop_config,
            theta_key=theta_key,
            condition_keys=condition_keys,
        )

        self.posterior_cls = posterior_cls
        self.embedding_config = embedding_config
        self.posterior_config = posterior_config or {}
        self.optimizer_config = optimizer_config
        self.inference_config = inference_config
        self.latent_mode = latent_mode

        # Device setup
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            debug(f"Auto-detected device: {self.device}")

        # Model (initialized lazily)
        self._model: Optional[nn.Module] = None
        self._best_model: Optional[nn.Module] = None
        self._best_loss: float = float("inf")

        # Stored tensors from first batch (for model rebuild on load)
        self._init_theta: Optional[torch.Tensor] = None
        self._init_conditions: Optional[Dict[str, torch.Tensor]] = None

        # Optimizer/scheduler (initialized lazily)
        self._optimizer: Optional[AdamW] = None
        self._scheduler: Optional[ReduceLROnPlateau] = None

    # ==================== Model Creation ====================

    def _build_model(self, batch) -> nn.Module:
        """Build model from first batch."""
        # Import here to avoid circular imports
        from falcon.contrib.embedded_posterior import EmbeddedPosterior
        from falcon.contrib.torch_embedding import instantiate_embedding

        # Extract and store tensors for reload
        self._init_theta = torch.from_numpy(batch[self.theta_key])
        self._init_conditions = {
            k: torch.from_numpy(batch[k]) for k in self.condition_keys if k in batch
        }
        return self._create_model(self._init_theta, self._init_conditions)

    def _create_model(self, theta: torch.Tensor, conditions: Dict[str, torch.Tensor]) -> nn.Module:
        """Create EmbeddedPosterior from theta and conditions tensors."""
        from falcon.contrib.embedded_posterior import EmbeddedPosterior
        from falcon.contrib.torch_embedding import instantiate_embedding

        debug("Building model...")

        # Infer dtype from theta (preserves numpy precision, e.g. float64)
        dtype = theta.dtype

        # Create embedding and infer condition_dim
        embedding = instantiate_embedding(self.embedding_config).to(self.device, dtype=dtype)
        embedding.eval()
        with torch.no_grad():
            conditions_device = {k: v.to(self.device, dtype=dtype) for k, v in conditions.items()}
            embedded = embedding(conditions_device)

        # Create posterior with inferred dimensions
        posterior = self.posterior_cls(
            param_dim=theta.shape[1],
            condition_dim=embedded.shape[1],
            **self.posterior_config,
        ).to(self.device, dtype=dtype)

        debug(f"Model built with dtype={dtype}.")
        return EmbeddedPosterior(embedding, posterior)

    # ==================== Loss Computation ====================

    def _compute_loss(self, batch) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute loss from batch."""
        # Extract and convert to tensors (dtype inferred from numpy arrays)
        theta = torch.from_numpy(batch[self.theta_key]).to(self.device)
        theta_logprob = torch.from_numpy(batch[f"{self.theta_key}.logprob"])
        conditions = {
            k: torch.from_numpy(batch[k]).to(self.device)
            for k in self.condition_keys if k in batch
        }

        # Transform theta to latent space if mode specified
        if self.latent_mode is not None:
            theta = self.simulator_instance.inverse(theta, mode=self.latent_mode)

        # Record sample IDs for history
        ts = time.time()
        self.history["train_ids"].extend((ts, id) for id in batch._ids.tolist())

        # Compute loss in latent space
        loss = self._model.loss(theta, conditions)

        # Discard low-probability samples
        if self.inference_config.discard_samples:
            with torch.no_grad():
                self._model.eval()
                log_prob = self._model.log_prob(theta, conditions).cpu()
            log_ratio = log_prob - theta_logprob
            discard_mask = log_ratio < self.inference_config.log_ratio_threshold
            batch.discard(discard_mask)

        return loss, {"loss": loss.item()}

    # ==================== Model Cloning ====================

    def _clone_model(self, model: nn.Module) -> nn.Module:
        """Create independent copy of model with cloned tensors."""
        cloned = copy.deepcopy(model)
        cloned.load_state_dict(
            {k: v.clone() for k, v in model.state_dict().items()}
        )
        return cloned

    def _update_best_model(self) -> None:
        """Update best model weights from current model."""
        self._best_model.load_state_dict(
            {k: v.clone() for k, v in self._model.state_dict().items()}
        )

    # ==================== Training Implementation ====================

    def _initialize_model(self, batch) -> None:
        """Initialize model, best model, optimizer, and scheduler."""
        debug("Initializing model...")

        # Build model
        self._model = self._build_model(batch)

        # Clone for best model
        self._best_model = self._clone_model(self._model)

        # Setup optimizer and scheduler
        cfg = self.optimizer_config
        self._optimizer = AdamW(self._model.parameters(), lr=cfg.lr, betas=cfg.betas)
        self._scheduler = ReduceLROnPlateau(
            self._optimizer,
            mode="min",
            factor=cfg.lr_decay_factor,
            patience=cfg.scheduler_patience,
        )

        self.networks_initialized = True
        debug("Model initialized.")

    def train_step(self, batch) -> Dict[str, float]:
        """Execute one training step with gradient update."""
        if not self.networks_initialized:
            self._initialize_model(batch)

        self._optimizer.zero_grad()
        self._model.train()
        loss, metrics = self._compute_loss(batch)
        loss.backward()
        self._optimizer.step()

        return metrics

    def val_step(self, batch) -> Dict[str, float]:
        """Execute one validation step without gradients."""
        with torch.no_grad():
            self._model.eval()
            _, metrics = self._compute_loss(batch)
        return metrics

    def on_epoch_end(self, epoch: int, val_metrics: Dict[str, float]) -> None:
        """Update best model and LR scheduler."""
        val_loss = val_metrics.get("loss", float("inf"))

        if val_loss < self._best_loss:
            self._best_loss = val_loss
            self._update_best_model()
            log({"checkpoint": epoch})

        self._scheduler.step(val_loss)
        log({"lr": self._optimizer.param_groups[0]["lr"]})

    # ==================== Inference Model Access ====================

    @property
    def inference_model(self) -> Optional[nn.Module]:
        """Get the best model for inference. Returns None if not yet trained."""
        return self._best_model

    @property
    def has_trained_model(self) -> bool:
        """Check if a trained model is available for inference."""
        return self._best_model is not None

    # ==================== Sampling Methods ====================

    def sample_prior(self, num_samples: int, conditions: Optional[Dict] = None) -> RVBatch:
        """Sample from the prior distribution."""
        if conditions:
            raise ValueError("Conditions are not supported for sample_prior.")
        samples = self.simulator_instance.simulate_batch(num_samples)
        logprob = np.zeros(num_samples)
        return RVBatch(samples, logprob=logprob)

    def _sample(self, num_samples: int, conditions: Optional[Dict], gamma: Optional[float]) -> RVBatch:
        """Internal sampling using inference model. Falls back to prior if not trained."""
        if not self.has_trained_model:
            return self.sample_prior(num_samples)

        assert conditions, "Conditions must be provided for sampling."

        conditions_device = {
            k: v.to(self.device).expand(num_samples, *v.shape[1:])
            for k, v in conditions.items()
        }

        model = self.inference_model
        with torch.no_grad():
            model.eval()
            samples = model.sample(conditions_device, gamma=gamma)
            logprob = model.log_prob(samples, conditions_device)

            # Transform samples from latent space back to theta space
            if self.latent_mode is not None:
                samples = self.simulator_instance.forward(samples, mode=self.latent_mode)

        return RVBatch(samples.cpu().numpy(), logprob=logprob.cpu().numpy())

    def sample_posterior(self, num_samples: int, conditions: Optional[Dict] = None) -> RVBatch:
        """Sample from the posterior distribution q(theta|x)."""
        return self._sample(num_samples, conditions, gamma=None)

    def sample_proposal(self, num_samples: int, conditions: Optional[Dict] = None) -> RVBatch:
        """Sample from widened proposal distribution for adaptive resampling."""
        result = self._sample(num_samples, conditions, gamma=self.inference_config.gamma)
        log({
            "sample_proposal:mean": result.value.mean(),
            "sample_proposal:std": result.value.std(),
            "sample_proposal:logprob": result.logprob.mean(),
        })
        return result

    # ==================== Save/Load ====================

    def save(self, node_dir) -> None:
        """Save model state."""
        node_dir = Path(node_dir)

        if not self.networks_initialized:
            raise RuntimeError("Cannot save: networks not initialized.")

        # Save model weights
        torch.save(self._best_model.state_dict(), node_dir / "model.pth")

        # Save init tensors for model rebuild
        torch.save({"theta": self._init_theta, "conditions": self._init_conditions},
                   node_dir / "init_tensors.pth")

        # Save history
        torch.save(self.history["train_ids"], node_dir / "train_id_history.pth")
        torch.save(self.history["val_ids"], node_dir / "validation_id_history.pth")
        torch.save(self.history["epochs"], node_dir / "epochs.pth")
        torch.save(self.history["train_loss"], node_dir / "loss_train_posterior.pth")
        torch.save(self.history["val_loss"], node_dir / "loss_val_posterior.pth")
        torch.save(self.history["n_samples"], node_dir / "n_samples_total.pth")
        torch.save(self.history["elapsed_min"], node_dir / "elapsed_minutes.pth")

    def load(self, node_dir) -> None:
        """Load model state."""
        node_dir = Path(node_dir)

        # Load init tensors and rebuild model
        data = torch.load(node_dir / "init_tensors.pth")
        self._init_theta = data["theta"]
        self._init_conditions = data["conditions"]
        self._model = self._create_model(self._init_theta, self._init_conditions)
        self._best_model = self._clone_model(self._model)

        # Setup optimizer and scheduler
        cfg = self.optimizer_config
        self._optimizer = AdamW(self._model.parameters(), lr=cfg.lr, betas=cfg.betas)
        self._scheduler = ReduceLROnPlateau(
            self._optimizer,
            mode="min",
            factor=cfg.lr_decay_factor,
            patience=cfg.scheduler_patience,
        )

        self.networks_initialized = True

        # Load weights
        self._best_model.load_state_dict(torch.load(node_dir / "model.pth"))
