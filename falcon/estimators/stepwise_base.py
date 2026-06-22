"""Stepwise estimator with epoch-based training loop."""

import asyncio
import time
from abc import abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch

from falcon.core.base_estimator import BaseEstimator
from falcon.core.logger import log, debug, info, warning, error


@dataclass
class TrainingLoopConfig:
    """Generic training loop parameters."""

    max_epochs: int = 100
    batch_size: int = 128
    early_stop_patience: int = 16
    cache_sync_every: int = 0  # 0 = sync every epoch, N = sync every N epochs
    max_cache_samples: int = 0  # 0 = cache all, >0 = cache random subset
    cache_on_device: bool = False  # True = cache training data on estimator's device (GPU)
    prior_epochs: int = 0


class StepwiseEstimator(BaseEstimator):
    """
    Estimator with epoch-based training loop.

    Provides concrete implementations for:
    - train() with epoch iteration and early stopping
    - interrupt

    Subclasses must implement:
    - train_step() / val_step() / on_epoch_end()
    - sample_prior/posterior/proposal
    - save/load
    """

    def setup(
        self,
        simulator_instance,
        theta_key: Optional[str] = None,
        condition_keys: Optional[List[str]] = None,
    ):
        """Initialise runtime state shared by all stepwise estimators."""
        self.simulator_instance = simulator_instance
        self.param_dim = simulator_instance.param_dim
        self.theta_key = theta_key
        self.condition_keys = condition_keys or []
        self._terminated = False
        self._total_epochs_trained: int = 0
        self.networks_initialized = False
        self.history = {
            "train_ids": [],
            "val_ids": [],
            "epochs": [],
            "train_loss": [],
            "val_loss": [],
            "n_samples": [],
            "elapsed_min": [],
        }

    # ==================== Utilities ====================

    @staticmethod
    def _to_tensor(x, device=None):
        """Convert numpy array or torch tensor to the target device."""
        if isinstance(x, torch.Tensor):
            return x if device is None else x.to(device)
        return torch.from_numpy(np.asarray(x)) if device is None else torch.from_numpy(np.asarray(x)).to(device)

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
    def on_epoch_end(self, epoch: int, val_metrics: Dict[str, float]) -> Optional[Dict[str, float]]:
        """
        Hook called at end of each epoch.

        Use for:
        - Updating best model weights
        - LR scheduler step
        - Custom checkpointing logic

        Args:
            epoch: Current epoch number (0-indexed)
            val_metrics: Validation metrics from this epoch

        Returns:
            Optional dict of extra metrics to include in epoch summary line.
        """
        pass

    # ==================== Concrete Methods ====================

    async def train(self, buffer) -> None:
        """Main training loop with epochs and early stopping."""
        keys = [f"{self.theta_key}.value", f"{self.theta_key}.log_prob",
                *[f"{k}.value" for k in self.condition_keys]]
        await self._train(buffer, keys)

    async def _train(self, buffer, keys) -> None:
        """Epoch-based training with CPU-cached dataloader."""
        sync_every = self.cache_sync_every if self.cache_sync_every > 0 else 1

        train_cache = buffer.cached_loader(keys, max_cache_samples=self.max_cache_samples)
        val_cache = buffer.cached_val_loader(keys, max_cache_samples=0)

        train_cache.sync()
        val_cache.sync()

        best_val_loss = float("inf")
        epochs_no_improve = 0
        total_steps = 0
        t0 = time.perf_counter()

        for epoch in range(self.max_epochs):
            log({"epoch": self._total_epochs_trained + 1})

            # Periodic incremental sync
            if epoch > 0 and epoch % sync_every == 0:
                train_cache.sync()
                val_cache.sync()

            # === Training phase ===
            steps_per_epoch = max(1, train_cache.count // self.batch_size)
            train_metrics_sum = {}
            num_train_batches = 0

            for step in range(steps_per_epoch):
                batch = train_cache.sample_batch(self.batch_size)
                metrics = self.train_step(batch)

                for k, v in metrics.items():
                    train_metrics_sum[k] = train_metrics_sum.get(k, 0) + v
                num_train_batches += 1
                total_steps += 1

                for k, v in metrics.items():
                    log({f"train:{k}": v})

                await asyncio.sleep(0)

            train_metrics_avg = {
                k: v / num_train_batches for k, v in train_metrics_sum.items()
            }

            # === Validation phase ===
            val_metrics_sum = {}
            num_val_samples = 0
            val_steps = max(1, val_cache.count // self.batch_size)

            for step in range(val_steps):
                batch = val_cache.sample_batch(self.batch_size)
                metrics = self.val_step(batch)
                bs = len(batch)

                for k, v in metrics.items():
                    val_metrics_sum[k] = val_metrics_sum.get(k, 0) + v * bs
                num_val_samples += bs

                await asyncio.sleep(0)

            val_metrics_avg = {
                k: v / num_val_samples for k, v in val_metrics_sum.items()
            }

            for k, v in val_metrics_avg.items():
                log({f"val:{k}": v})

            extra_metrics = self.on_epoch_end(epoch, val_metrics_avg)

            # Log step/sample counts
            log({"total_steps": total_steps})
            try:
                n_sims = buffer.get_stats()["total_length"]
                log({"n_samples": n_sims})
            except Exception:
                n_sims = None

            # Print epoch summary
            train_loss = train_metrics_avg.get("loss", float("nan"))
            val_loss = val_metrics_avg.get("loss", float("inf"))
            summary = (
                f"Epoch {self._total_epochs_trained + 1}/{self.max_epochs}"
                f" | steps={total_steps}"
            )
            if n_sims is not None:
                summary += f" | n_sims={n_sims}"
            summary += (
                f" | train_loss={train_loss:.3e}"
                f" | val_loss={val_loss:.3e}"
            )
            if extra_metrics:
                for k, v in extra_metrics.items():
                    summary += f" | {k}={v:.3e}"
            info(summary)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            self._record_epoch_history(
                epoch, train_metrics_avg, val_metrics_avg, t0, buffer
            )
            self._total_epochs_trained += 1

            if epochs_no_improve >= self.early_stop_patience:
                info("Early stopping triggered.")
                break

            if self._terminated:
                break

    def _record_epoch_history(
        self, epoch, train_metrics, val_metrics, t0, buffer
    ) -> None:
        """Record metrics to history dict."""
        self.history["epochs"].append(self._total_epochs_trained + 1)
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

    def interrupt(self) -> None:
        """Terminate training loop."""
        self._terminated = True

