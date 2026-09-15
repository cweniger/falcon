"""Stepwise estimator with round-based training loop."""

import asyncio
import copy
import math
import time
from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from falcon.core.base_estimator import BaseEstimator
from falcon.core.logger import log, debug, info, warning, error


@dataclass
class NetworkGroup:
    """Networks that are snapshotted, compared and promoted together.

    Attributes:
        current: Density network being trained this round.
        best: Best density network so far; used for proposals and saved.
        metric: Validation metric that decides whether ``current`` replaces ``best``.
        current_embedding: Embedding feeding ``current``, if any.
        best_embedding: Embedding feeding ``best``, if any.
    """

    current: nn.Module
    best: nn.Module
    metric: str
    current_embedding: Optional[nn.Module] = None
    best_embedding: Optional[nn.Module] = None

    def current_modules(self) -> List[nn.Module]:
        return [m for m in (self.current, self.current_embedding) if m is not None]

    def best_modules(self) -> List[nn.Module]:
        return [m for m in (self.best, self.best_embedding) if m is not None]


class StepwiseEstimator(BaseEstimator):
    """
    Estimator with a round-based training loop.

    Training runs in rounds (see ``docs/training.md`` for the terminology):

    1. Refresh the training and validation caches from the buffer; the data
       then stays fixed for the whole round.
    2. Start from the best networks and train for up to ``max_epochs`` epochs
       (passes over the training set), validating every ``val_every_epochs``
       epochs and at the last epoch. The round ends early once the best
       validation loss is ``patience_epochs`` epochs old.
    3. Restore each network group to its best validated epoch of the round and
       compare it with the best network on the round's validation set. Groups
       that improve are promoted; the round is accepted if the first (primary)
       group was promoted.
    4. After an accepted round, run one discard sweep over the training and
       validation sets with the new best networks.

    Training stops after ``patience_rounds`` rejected rounds in a row, after
    ``max_rounds`` rounds, or when interrupted. Proposal sampling can run
    between any two training steps; it uses the best networks, which only
    change when a round is accepted.

    Subclasses must implement:
    - train_step() / val_step() / on_validation_end() / discard_mask()
    - _network_groups()
    - sample_prior/posterior/proposal
    - save/load

    The loop reads its parameters off ``self`` -- max_rounds, patience_rounds,
    max_epochs, patience_epochs, val_every_epochs, batch_size,
    max_cache_samples, cache_on_device, discard_samples -- which each subclass
    sets from its own __init__ kwargs.
    """

    def setup(
        self,
        simulator_instance,
        theta_key: Optional[str] = None,
        condition_keys: Optional[List[str]] = None,
    ):
        """Initialise runtime state shared by all stepwise estimators."""
        if self.max_epochs < 1 or self.val_every_epochs < 1:
            raise ValueError(
                f"max_epochs={self.max_epochs} and val_every_epochs={self.val_every_epochs} must be >= 1"
            )
        self.simulator_instance = simulator_instance
        self.param_dim = simulator_instance.param_dim
        self.theta_key = theta_key
        self.condition_keys = condition_keys or []
        self._terminated = False
        self._total_epochs_trained: int = 0
        self._total_steps: int = 0
        self.networks_initialized = False

        # Round state
        self._round: int = 0            # rounds started
        self._round_epoch: int = 0      # epoch within the current round
        self._rounds_accepted: int = 0
        self._stall: int = 0            # consecutive rejected rounds
        self._has_best: bool = False
        self.best_val_loss: Optional[float] = None  # best network, latest acceptance test

        self.history = {
            "train_ids": [],
            "val_ids": [],
            "epochs": [],
            "train_loss": [],
            "val_loss": [],
            "n_samples": [],
            "elapsed_min": [],
            "rounds": [],
        }

    # ==================== Utilities ====================

    @staticmethod
    def _to_tensor(x, device=None):
        """Convert numpy array or torch tensor to the target device."""
        if isinstance(x, torch.Tensor):
            return x if device is None else x.to(device)
        return torch.from_numpy(np.asarray(x)) if device is None else torch.from_numpy(np.asarray(x)).to(device)

    def _summary(self, batch, group: str, conditions, use_best: bool = False, train: bool = False):
        """Embedding output for ``conditions``, using the embedding of a network group.

        Training and validation route all embedding calls through here.

        Args:
            batch: Batch the conditions were taken from.
            group: Name of the network group in ``_network_groups()``.
            conditions: Dict of condition tensors, as passed to the embedding.
            use_best: Use the best network's embedding instead of the current one.
            train: Run the embedding in train mode.
        """
        g = self._network_groups()[group]
        embedding = g.best_embedding if use_best else g.current_embedding
        embedding.train(train)
        return embedding(conditions)

    def _epochs_to_validations(self, epochs: int) -> int:
        """Number of validations spanning ``epochs`` epochs (rounded up)."""
        return math.ceil(epochs / self.val_every_epochs)

    def _use_prior_proposal(self) -> bool:
        """Whether proposals should still come from the prior."""
        return not self._has_best or self._round <= self.prior_rounds

    @staticmethod
    def _copy_modules(src: List[nn.Module], dst: List[nn.Module]) -> None:
        for s, d in zip(src, dst):
            d.load_state_dict(s.state_dict())

    @staticmethod
    def _snapshot(modules: List[nn.Module]) -> list:
        return [copy.deepcopy(m.state_dict()) for m in modules]

    @staticmethod
    def _restore(modules: List[nn.Module], states: list) -> None:
        for m, s in zip(modules, states):
            m.load_state_dict(s)

    def _save_round_state(self, node_dir: Path) -> None:
        torch.save(
            {"rounds": self._round, "accepted": self._rounds_accepted, "stall": self._stall},
            Path(node_dir) / "round_state.pth",
        )

    def _load_round_state(self, node_dir: Path) -> None:
        """Restore round counters; the loaded networks count as the best networks.

        A resumed run continues the counts: ``max_rounds`` and ``patience_rounds``
        include the rounds trained before the resume.
        """
        path = Path(node_dir) / "round_state.pth"
        if path.exists():
            state = torch.load(path)
            self._round = state["rounds"]
            self._rounds_accepted = state["accepted"]
            self._stall = state.get("stall", 0)
        self._has_best = True

    # ==================== Abstract Methods ====================

    @abstractmethod
    def train_step(self, batch) -> Dict[str, float]:
        """
        Execute one training step on the current networks.

        Args:
            batch: Batch object containing tensors accessible via batch[key]

        Returns:
            Dict of metrics to log. Must include "loss" key.

        Note:
            No batch.discard() calls; discarding happens in the sweep after an
            accepted round (see discard_mask).
        """
        pass

    @abstractmethod
    def val_step(self, batch, use_best: bool = False) -> Dict[str, float]:
        """
        Evaluate one validation batch. Runs under ``torch.no_grad()``.

        Args:
            batch: Batch object containing tensors
            use_best: Evaluate the best networks instead of the current ones.

        Returns:
            Dict of metrics. Must include "loss" (early stopping) and the
            ``metric`` of every network group (acceptance test).

        Note:
            No parameter updates and no batch.discard() calls.
        """
        pass

    @abstractmethod
    def on_validation_end(self, epoch: int, val_metrics: Dict[str, float]) -> Optional[Dict[str, float]]:
        """
        Hook called after each validation within a round (e.g. LR scheduler step).

        Args:
            epoch: Epoch within the round (1-indexed)
            val_metrics: Metrics of this validation

        Returns:
            Optional dict of extra metrics to include in the epoch summary line.
        """
        pass

    @abstractmethod
    def discard_mask(self, batch):
        """
        Discard test for one batch, evaluated with the best networks.

        Called during the discard sweep after an accepted round, for training
        and validation samples alike. Runs under ``torch.no_grad()``.

        Returns:
            Boolean mask (tensor or array) of length len(batch); True = discard.
        """
        pass

    @abstractmethod
    def _network_groups(self) -> Dict[str, NetworkGroup]:
        """Network groups of this estimator; the first entry is the primary group."""
        pass

    def on_round_start(self) -> None:
        """Hook called at the start of each round, after the best networks were
        copied into the current ones (e.g. reset the learning rate)."""
        pass

    # ==================== Training Loop ====================

    async def train(self, buffer) -> None:
        """Main training loop: rounds of epochs on fixed data."""
        keys = [f"{self.theta_key}.value", f"{self.theta_key}.log_prob",
                *[f"{k}.value" for k in self.condition_keys]]
        await self._train(buffer, keys)

    async def _train(self, buffer, keys) -> None:
        """Round-based training with cached dataloaders."""
        train_cache = buffer.cached_loader(keys, max_cache_samples=self.max_cache_samples)
        val_cache = buffer.cached_val_loader(keys, max_cache_samples=0)

        if self.patience_epochs % self.val_every_epochs:
            effective = self._epochs_to_validations(self.patience_epochs) * self.val_every_epochs
            warning(
                f"patience_epochs={self.patience_epochs} is not a multiple of "
                f"val_every_epochs={self.val_every_epochs}; rounds end after {effective} "
                "epochs without improvement"
            )

        t0 = time.perf_counter()
        while not self._terminated:
            await train_cache.refresh()
            await val_cache.refresh()
            if self._terminated:
                break  # stopped while the data was being fetched: don't start a round
            if train_cache.count == 0 or val_cache.count == 0:
                warning(
                    f"Waiting for data: {train_cache.count} training and "
                    f"{val_cache.count} validation samples cached"
                )
                await asyncio.sleep(1.0)
                continue

            self._round += 1
            self._begin_round()
            epochs = await self._train_epochs(train_cache, val_cache, buffer, t0)
            if epochs == 0:
                break  # interrupted before the first validation: nothing to test

            accepted, record = await self._accept_round(val_cache)
            record.update(epochs=epochs, n_train=train_cache.count, n_val=val_cache.count)

            if accepted:
                self._stall = 0
                if self.discard_samples and not self._terminated:  # graceful stop: no sweep
                    await train_cache.refresh()
                    await val_cache.refresh()
                    record.update(await self._discard_sweep(train_cache, val_cache))
            else:
                self._stall += 1
            self._log_round(record)

            if self._terminated:
                break
            if self._stall >= self.patience_rounds:
                info(f"Training finished: {self._stall} rejected rounds in a row.")
                break
            if self.max_rounds and self._round >= self.max_rounds:
                info(f"Training finished: reached max_rounds={self.max_rounds}.")
                break

    def _begin_round(self) -> None:
        """Start the round from the best networks."""
        self._round_epoch = 0
        if not self.networks_initialized:
            return  # networks are built on the first training step
        if self._has_best:
            for group in self._network_groups().values():
                self._copy_modules(group.best_modules(), group.current_modules())
        self.on_round_start()

    async def _train_epochs(self, train_cache, val_cache, buffer, t0) -> int:
        """Train the current networks for one round.

        Restores each network group to its best validated epoch before returning.

        Returns:
            Number of the last validated epoch; 0 if no validation completed.
        """
        round_best = {}   # group -> best metric value so far this round
        snapshots = {}    # group -> state dicts at that validation
        best_loss, best_epoch = float("inf"), 0
        last_validated = 0

        for epoch in range(1, self.max_epochs + 1):
            self._round_epoch = epoch
            train_metrics = await self._train_epoch(train_cache)
            if train_metrics is None:
                break  # interrupted; progress since the last validation is dropped
            self._total_epochs_trained += 1

            if epoch % self.val_every_epochs and epoch != self.max_epochs:
                self._log_epoch(epoch, train_metrics, None, None, t0, buffer)
                continue

            val_metrics = await self._validate(val_cache)
            if val_metrics is None:
                break
            last_validated = epoch
            for name, group in self._network_groups().items():
                value = val_metrics.get(group.metric, float("nan"))
                if value < round_best.get(name, float("inf")):
                    round_best[name] = value
                    snapshots[name] = self._snapshot(group.current_modules())
                    log({f"checkpoint:{name}": epoch})
            extra = self.on_validation_end(epoch, val_metrics)
            self._log_epoch(epoch, train_metrics, val_metrics, extra, t0, buffer)

            val_loss = val_metrics.get("loss", float("nan"))
            if val_loss < best_loss:
                best_loss, best_epoch = val_loss, epoch
            elif epoch - best_epoch >= self.patience_epochs:
                info(f"Round {self._round} converged: no improvement since epoch {best_epoch}.")
                break

        if last_validated:
            for name, group in self._network_groups().items():
                if name in snapshots:
                    self._restore(group.current_modules(), snapshots[name])
        return last_validated

    async def _train_epoch(self, train_cache) -> Optional[Dict[str, float]]:
        """One shuffled pass over the training set; None if interrupted."""
        sums, num_batches = {}, 0
        for batch in train_cache.iter_batches(self.batch_size, shuffle=True, drop_last=True):
            metrics = self.train_step(batch)
            for k, v in metrics.items():
                sums[k] = sums.get(k, 0.0) + v
                log({f"train:{k}": v})
            num_batches += 1
            self._total_steps += 1
            await asyncio.sleep(0)
            if self._terminated:
                return None
        return {k: v / num_batches for k, v in sums.items()}

    async def _validate(self, val_cache, use_best: bool = False, interruptible: bool = True):
        """One full pass over the validation set; None if interrupted."""
        sums, num_samples = {}, 0
        for batch in val_cache.iter_batches(self.batch_size):
            with torch.no_grad():
                metrics = self.val_step(batch, use_best=use_best)
            for k, v in metrics.items():
                sums[k] = sums.get(k, 0.0) + v * len(batch)
            num_samples += len(batch)
            await asyncio.sleep(0)
            if interruptible and self._terminated:
                return None
        return {k: v / num_samples for k, v in sums.items()}

    async def _accept_round(self, val_cache):
        """Compare the round's candidate with the best networks and promote winners.

        Returns:
            (accepted, record): whether the primary group was promoted, and a
            dict describing the decision for logging.
        """
        groups = self._network_groups()
        candidate = await self._validate(val_cache, use_best=False, interruptible=False)
        best = await self._validate(val_cache, use_best=True, interruptible=False) if self._has_best else None

        record = {"round": self._round, "groups": {}}
        for name, group in groups.items():
            cand_value = candidate.get(group.metric, float("nan"))
            best_value = best.get(group.metric, float("nan")) if best is not None else None
            promoted = best is None or cand_value < best_value
            if promoted:
                self._copy_modules(group.current_modules(), group.best_modules())
            record["groups"][name] = {"candidate": cand_value, "best": best_value, "promoted": promoted}

        primary = next(iter(groups))
        accepted = record["groups"][primary]["promoted"]
        record["accepted"] = accepted
        if accepted:
            self._rounds_accepted += 1
        self._has_best = True
        primary_values = record["groups"][primary]
        self.best_val_loss = primary_values["candidate"] if accepted else primary_values["best"]
        return accepted, record

    async def _discard_sweep(self, train_cache, val_cache) -> Dict[str, int]:
        """Run the discard test once over every training and validation sample."""
        counts, flagged = {"n_discarded_train": 0, "n_discarded_val": 0}, []
        for name, cache in (("train", train_cache), ("val", val_cache)):
            for batch in cache.iter_batches(self.batch_size):
                with torch.no_grad():
                    mask = self.discard_mask(batch)
                mask = mask.cpu().numpy() if hasattr(mask, "cpu") else np.asarray(mask)
                ids = batch._ids[mask.astype(bool)]
                flagged.append(ids)
                counts[f"n_discarded_{name}"] += len(ids)
                await asyncio.sleep(0)
                if self._terminated:
                    break
            if self._terminated:
                break  # the samples tested so far are still discarded
        ids = np.concatenate(flagged) if flagged else np.array([], dtype=int)
        if len(ids) > 0:
            train_cache.dataset_manager.deactivate.remote(ids.tolist())
        return counts

    # ==================== Logging ====================

    def _log_epoch(self, epoch, train_metrics, val_metrics, extra, t0, buffer) -> None:
        """Log and print one epoch; validation fields only when it was validated."""
        log({"round": self._round, "epoch": epoch, "total_steps": self._total_steps})
        if val_metrics is not None:
            for k, v in val_metrics.items():
                log({f"val:{k}": v})
        try:
            n_sims = buffer.get_stats()["total_length"]
            log({"n_samples": n_sims})
        except Exception:
            n_sims = None

        summary = f"Round {self._round} | epoch {epoch}/{self.max_epochs} | steps={self._total_steps}"
        if n_sims is not None:
            summary += f" | n_sims={n_sims}"
        summary += f" | train_loss={train_metrics.get('loss', float('nan')):.3e}"
        if val_metrics is not None:
            summary += f" | val_loss={val_metrics.get('loss', float('nan')):.3e}"
        for k, v in (extra or {}).items():
            summary += f" | {k}={v:.3e}"
        info(summary)

        self.history["epochs"].append(self._total_epochs_trained)
        self.history["train_loss"].append(train_metrics.get("loss", float("nan")))
        self.history["val_loss"].append(
            val_metrics.get("loss", float("nan")) if val_metrics is not None else float("nan")
        )
        elapsed = (time.perf_counter() - t0) / 60.0
        self.history["elapsed_min"].append(elapsed)
        log({"elapsed_minutes": elapsed})
        if n_sims is not None:
            self.history["n_samples"].append(n_sims)

    def _log_round(self, record) -> None:
        """Log and print the outcome of a round."""
        metrics = {
            "round": record["round"],
            "round:epochs": record["epochs"],
            "round:accepted": int(record["accepted"]),
            "round:n_train": record["n_train"],
            "round:n_val": record["n_val"],
        }
        parts = [
            f"Round {record['round']} {'ACCEPTED' if record['accepted'] else 'rejected'}",
            f"epochs={record['epochs']}",
            f"n_train={record['n_train']} n_val={record['n_val']}",
        ]
        for name, g in record["groups"].items():
            metrics[f"round:{name}:candidate"] = g["candidate"]
            metrics[f"round:{name}:promoted"] = int(g["promoted"])
            best = "none" if g["best"] is None else f"{g['best']:.3e}"
            if g["best"] is not None:
                metrics[f"round:{name}:best"] = g["best"]
            parts.append(
                f"{name}: {g['candidate']:.3e} vs best {best} "
                f"{'promoted' if g['promoted'] else 'kept'}"
            )
        for key in ("n_discarded_train", "n_discarded_val"):
            if key in record:
                metrics[f"round:{key}"] = record[key]
        if "n_discarded_train" in record:
            parts.append(f"discarded train={record['n_discarded_train']} val={record['n_discarded_val']}")
        if not record["accepted"]:
            parts.append(f"rejected in a row={self._stall}/{self.patience_rounds}")
        log(metrics)
        info(" | ".join(parts))
        self.history["rounds"].append(record)

    def interrupt(self) -> None:
        """Stop training at the next step; the round's acceptance test still runs."""
        self._terminated = True
