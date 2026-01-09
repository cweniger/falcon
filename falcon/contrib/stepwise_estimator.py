"""Stepwise estimator with epoch-based training loop."""

import asyncio
import time
from abc import abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional

from falcon.core.base_estimator import BaseEstimator
from falcon.core.logging import log, info


@dataclass
class TrainingLoopConfig:
    """Generic training loop parameters."""

    num_epochs: int = 100
    batch_size: int = 128
    early_stop_patience: int = 16
    reset_network_after_pause: bool = False


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
