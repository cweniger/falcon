"""Round-based training loop, independent of the network framework."""

import json
import math
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np

from falcon.core.base_estimator import BaseEstimator, StateTree
from falcon.core.logger import log, info, warning
from falcon.core.state_io import read_checkpoint, write_checkpoint

HISTORY_NAME = "training_history.npz"


class RoundTrainer:
    """
    Trains a model in rounds (see ``docs/training.md`` for the terminology).

    1. Refresh the training and validation caches from the buffer; the data
       then stays fixed for the whole round.
    2. Restore the best state and validate it once: this is the best
       network's loss on the round's validation set.
    3. Hand the training set to the model (``on_train_start``), then train
       for up to ``max_epochs`` epochs, validating every ``val_every_epochs``
       epochs and at the last epoch. The round ends early once the best
       validation loss is ``patience_epochs`` epochs old.
    4. Restore each network group to its best validated epoch of the round,
       validate it, and promote the groups that beat the best network. The
       round is accepted if the primary group was promoted.
    5. Install the best state and let the model update its proposal state
       (``on_round_end``).
    6. Publish the new best state and proposal state, then, after an
       accepted round, run one discard sweep over the training and
       validation sets.

    Training stops after ``patience_rounds`` rejected rounds in a row, after
    ``max_rounds`` rounds, or on ``request_stop()``.

    The trainer owns everything that is not network state: round counters,
    the best state (as numpy trees), history and the checkpoint. A model's
    proposal state (``export_proposal``) belongs to the model; the trainer
    only publishes and saves it.

    Args:
        model: The model to train (``BaseEstimator``).
        publish: Callable(tree) that hands a state to the samplers and returns
            once they have installed it. Called with the round counters at the
            start of every round, with the full state after a promotion, and
            with the proposal state whenever it changed.
        meta: Extra entries for the published ``meta`` (e.g. node keys).
    """

    def __init__(self, model: BaseEstimator, publish: Optional[Callable[[StateTree], Any]] = None,
                 meta: Optional[Dict[str, Any]] = None):
        self.model = model
        self.config = model.round_config
        self.publish = publish or (lambda tree: None)
        self.extra_meta = dict(meta or {})
        if self.config.max_epochs < 1 or self.config.val_every_epochs < 1:
            raise ValueError(
                f"max_epochs={self.config.max_epochs} and "
                f"val_every_epochs={self.config.val_every_epochs} must be >= 1"
            )

        self.status = "idle"
        self._stop_requested = False

        # Round state
        self.round = 0              # rounds started
        self.round_epoch = 0        # epoch within the current round
        self.rounds_accepted = 0
        self.best_round = 0         # last round in which a group was promoted
        self.stall = 0              # consecutive rejected rounds
        self.total_epochs = 0
        self.total_steps = 0
        self.best_val_loss: Optional[float] = None  # best network, latest acceptance test

        # Best state, as published
        self.init: Optional[StateTree] = None
        self.best: Dict[str, StateTree] = {}
        self._init_published = False

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

    # ==================== Control ====================

    def request_stop(self) -> None:
        """Stop training after the current step; the round's acceptance test still runs."""
        self._stop_requested = True

    @property
    def stopped(self) -> bool:
        return self._stop_requested

    @property
    def has_best(self) -> bool:
        return bool(self.best)

    def _epochs_to_validations(self, epochs: int) -> int:
        """Number of validations spanning ``epochs`` epochs (rounded up)."""
        return math.ceil(epochs / self.config.val_every_epochs)

    # ==================== Published state ====================

    def meta(self) -> Dict[str, Any]:
        return {
            **self.extra_meta,
            "round": self.round,
            "has_best": self.has_best,
            "best_round": self.best_round,
            "rounds_accepted": self.rounds_accepted,
            "stall": self.stall,
            "total_epochs": self.total_epochs,
        }

    def state(self) -> StateTree:
        """The best state in the checkpoint format."""
        tree: StateTree = {"meta": self.meta()}
        if self.has_best:
            tree["groups"] = dict(self.best)
            tree["init"] = self.init
        proposal = self.model.export_proposal(full=True)
        if proposal is not None:
            tree["proposal"] = proposal
        return tree

    def _publish(self, weights: bool, proposal: bool = False) -> None:
        tree: StateTree = {"meta": self.meta()}
        if weights:
            tree["groups"] = dict(self.best)
            # The samplers build their networks once, so init is sent only the first time
            if not self._init_published:
                tree["init"] = self.init
            self._init_published = True
        if proposal:
            tree["proposal"] = self.model.export_proposal(full=False)
        self.publish(tree)

    # ==================== Training loop ====================

    def run(self, buffer) -> None:
        """Train until a stopping criterion is met."""
        cfg = self.config
        keys = self.model.batch_keys()
        train_cache = buffer.cached_loader(keys, max_cache_samples=cfg.max_cache_samples)
        val_cache = buffer.cached_val_loader(keys, max_cache_samples=0)

        if cfg.patience_epochs % cfg.val_every_epochs:
            effective = self._epochs_to_validations(cfg.patience_epochs) * cfg.val_every_epochs
            warning(
                f"patience_epochs={cfg.patience_epochs} is not a multiple of "
                f"val_every_epochs={cfg.val_every_epochs}; rounds end after {effective} "
                "epochs without improvement"
            )

        self.status = "training"
        t0 = time.perf_counter()
        while not self.stopped:
            train_cache.refresh()
            val_cache.refresh()
            if self.stopped:
                break  # stopped while the data was being fetched: don't start a round
            if train_cache.count == 0 or val_cache.count == 0:
                warning(
                    f"Waiting for data: {train_cache.count} training and "
                    f"{val_cache.count} validation samples cached"
                )
                time.sleep(1.0)
                continue

            self.round += 1
            train_batches = max(1, train_cache.count // cfg.batch_size)
            val_batches = -(-val_cache.count // cfg.batch_size)
            info(
                f"Round {self.round} starting | n_train={train_cache.count} "
                f"({train_batches} batches of {cfg.batch_size}) | "
                f"n_val={val_cache.count} ({val_batches} batches)"
            )
            self._begin_round(train_cache)
            self._publish(weights=False)

            baseline = None
            if self.has_best:
                baseline = self._validate(val_cache)
                if baseline is None:
                    break  # stopped before training

            self.model.on_train_start(train_cache.iter_batches(cfg.batch_size))
            epochs = self._train_epochs(train_cache, val_cache, t0)
            if epochs == 0:
                break  # interrupted before the first validation: nothing to test

            accepted, promoted, record = self._accept_round(val_cache, baseline)
            record.update(epochs=epochs, n_train=train_cache.count, n_val=val_cache.count)
            proposal = False
            if not self.stopped:  # graceful stop: the proposal is not moved
                proposal = self._end_round(record)
            if promoted or proposal:
                self._publish(weights=promoted, proposal=proposal)

            if accepted:
                self.stall = 0
                if cfg.discard_samples and not self.stopped:  # graceful stop: no sweep
                    train_cache.refresh()
                    val_cache.refresh()
                    record.update(self._discard_sweep(train_cache, val_cache))
            else:
                self.stall += 1
            self._log_round(record, buffer)

            if self.stopped:
                break
            if self.stall >= cfg.patience_rounds:
                info(f"Training finished: {self.stall} rejected rounds in a row.")
                break
            if cfg.max_rounds and self.round >= cfg.max_rounds:
                info(f"Training finished: reached max_rounds={cfg.max_rounds}.")
                break
        self.status = "done"

    def _begin_round(self, train_cache) -> None:
        """Start the round from the best state; build the networks in the first round."""
        self.round_epoch = 0
        if not self.model.built:
            batch = next(train_cache.iter_batches(self.config.batch_size))
            self.init = self.model.init_from_batch(batch)
            self.model.build(self.init)
            return
        for name, tree in self.best.items():
            self.model.import_state(name, tree)
        self.model.on_round_start()

    def _end_round(self, record) -> bool:
        """Install the best state and let the model update its proposal; True if it changed."""
        for name, tree in self.best.items():
            self.model.import_state(name, tree)
        promoted = {name: g["promoted"] for name, g in record["groups"].items()}
        return bool(self.model.on_round_end(promoted))

    def _train_epochs(self, train_cache, val_cache, t0) -> int:
        """Train the current networks for one round.

        Restores each network group to its best validated epoch before returning.

        Returns:
            Number of the last validated epoch; 0 if no validation completed.
        """
        cfg = self.config
        groups = self.model.groups()
        round_best = {}   # group -> best metric value so far this round
        snapshots = {}    # group -> snapshot at that validation
        best_loss, best_epoch = float("inf"), 0
        last_validated = 0

        for epoch in range(1, cfg.max_epochs + 1):
            self.round_epoch = epoch
            train_metrics = self._train_epoch(train_cache)
            if train_metrics is None:
                break  # interrupted; progress since the last validation is dropped
            self.total_epochs += 1

            if epoch % cfg.val_every_epochs and epoch != cfg.max_epochs:
                self._log_epoch(epoch, train_metrics, None, None, t0)
                continue

            val_metrics = self._validate(val_cache)
            if val_metrics is None:
                break
            last_validated = epoch
            for name, metric in groups.items():
                value = val_metrics.get(metric, float("nan"))
                if value < round_best.get(name, float("inf")):
                    round_best[name] = value
                    snapshots[name] = self.model.snapshot(name)
                    log({f"checkpoint:{name}": epoch})
            extra = self.model.on_validation_end(epoch, val_metrics)
            self._log_epoch(epoch, train_metrics, val_metrics, extra, t0)

            val_loss = val_metrics.get("loss", float("nan"))
            if val_loss < best_loss:
                best_loss, best_epoch = val_loss, epoch
            elif epoch - best_epoch >= cfg.patience_epochs:
                info(f"Round {self.round} converged: no improvement since epoch {best_epoch}.")
                break

        if last_validated:
            for name, snapshot in snapshots.items():
                self.model.restore(name, snapshot)
        return last_validated

    def _train_epoch(self, train_cache) -> Optional[Dict[str, float]]:
        """One shuffled pass over the training set; None if interrupted."""
        sums, num_batches = {}, 0
        for batch in train_cache.iter_batches(self.config.batch_size, shuffle=True, drop_last=True):
            self._record_ids("train_ids", batch)
            metrics = self.model.train_step(batch)
            for k, v in metrics.items():
                sums[k] = sums.get(k, 0.0) + v
                log({f"train:{k}": v})
            num_batches += 1
            self.total_steps += 1
            if self.stopped:
                return None
        return {k: v / num_batches for k, v in sums.items()}

    def _validate(self, val_cache, interruptible: bool = True):
        """One full pass over the validation set; None if interrupted."""
        sums, num_samples = {}, 0
        for batch in val_cache.iter_batches(self.config.batch_size):
            self._record_ids("val_ids", batch)
            metrics = self.model.evaluate(batch)
            for k, v in metrics.items():
                sums[k] = sums.get(k, 0.0) + v * len(batch)
            num_samples += len(batch)
            if interruptible and self.stopped:
                return None
        return {k: v / num_samples for k, v in sums.items()}

    def _record_ids(self, key: str, batch) -> None:
        ts = time.time()
        self.history[key].extend((ts, i) for i in batch._ids.tolist())

    def _accept_round(self, val_cache, baseline):
        """Validate the round's candidate and promote the groups that beat the best.

        Args:
            baseline: Validation metrics of the best state on this round's data,
                or None if there is no best state yet.

        Returns:
            (accepted, promoted, record): whether the primary group was promoted,
            whether any group was, and a dict describing the decision.
        """
        groups = self.model.groups()
        candidate = self._validate(val_cache, interruptible=False)

        record = {"round": self.round, "groups": {}}
        promoted_any = False
        for name, metric in groups.items():
            cand_value = candidate.get(metric, float("nan"))
            best_value = baseline.get(metric, float("nan")) if baseline is not None else None
            promoted = baseline is None or cand_value < best_value
            if promoted:
                self.best[name] = self.model.export_state(name)
                self.best_round = self.round
                promoted_any = True
            record["groups"][name] = {"candidate": cand_value, "best": best_value, "promoted": promoted}

        primary = next(iter(groups))
        accepted = record["groups"][primary]["promoted"]
        record["accepted"] = accepted
        if accepted:
            self.rounds_accepted += 1
        primary_values = record["groups"][primary]
        self.best_val_loss = primary_values["candidate"] if accepted else primary_values["best"]
        return accepted, promoted_any, record

    def _discard_sweep(self, train_cache, val_cache) -> Dict[str, int]:
        """Run the discard test once over every training and validation sample."""
        counts, flagged = {"n_discarded_train": 0, "n_discarded_val": 0}, []
        for name, cache in (("train", train_cache), ("val", val_cache)):
            for batch in cache.iter_batches(self.config.batch_size):
                mask = np.asarray(self.model.discard_mask(batch)).astype(bool)
                ids = batch._ids[mask]
                flagged.append(ids)
                counts[f"n_discarded_{name}"] += len(ids)
                if self.stopped:
                    break
            if self.stopped:
                break  # the samples tested so far are still discarded
        ids = np.concatenate(flagged) if flagged else np.array([], dtype=int)
        if len(ids) > 0:
            train_cache.dataset_manager.deactivate.remote(ids.tolist())
        return counts

    # ==================== Logging ====================

    def _log_epoch(self, epoch, train_metrics, val_metrics, extra, t0) -> None:
        """Log and print one epoch; validation fields only when it was validated."""
        log({"round": self.round, "epoch": epoch, "total_steps": self.total_steps})
        if val_metrics is not None:
            for k, v in val_metrics.items():
                log({f"val:{k}": v})

        summary = f"Round {self.round} | epoch {epoch}/{self.config.max_epochs} | steps={self.total_steps}"
        summary += f" | train_loss={train_metrics.get('loss', float('nan')):.3e}"
        if val_metrics is not None:
            summary += f" | val_loss={val_metrics.get('loss', float('nan')):.3e}"
        for k, v in (extra or {}).items():
            summary += f" | {k}={v:.3e}"
        info(summary)

        self.history["epochs"].append(self.total_epochs)
        self.history["train_loss"].append(train_metrics.get("loss", float("nan")))
        self.history["val_loss"].append(
            val_metrics.get("loss", float("nan")) if val_metrics is not None else float("nan")
        )
        elapsed = (time.perf_counter() - t0) / 60.0
        self.history["elapsed_min"].append(elapsed)
        log({"elapsed_minutes": elapsed})

    def _log_round(self, record, buffer) -> None:
        """Log and print the outcome of a round."""
        # Total samples ever simulated; fetched once per round, not per epoch,
        # because it is a blocking call to the dataset manager.
        try:
            n_sims = buffer.get_stats()["total_length"]
            log({"n_samples": n_sims})
            self.history["n_samples"].append(n_sims)
        except Exception:
            n_sims = None

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
        if n_sims is not None:
            parts.append(f"n_sims={n_sims}")
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
            parts.append(f"rejected in a row={self.stall}/{self.config.patience_rounds}")
        log(metrics)
        info(" | ".join(parts))
        self.history["rounds"].append(record)

    # ==================== Save / Load ====================

    def save(self, node_dir) -> bool:
        """Write the best state and the history; False if there is nothing to save."""
        node_dir = Path(node_dir)
        if not self.has_best:
            warning(f"Nothing to save in {node_dir}: no round has been completed")
            return False
        write_checkpoint(node_dir, self.state())
        self._save_history(node_dir / HISTORY_NAME)
        return True

    def load(self, node_dir) -> bool:
        """Resume from a checkpoint; False if there is none.

        A resumed run continues the counts: ``max_rounds`` and ``patience_rounds``
        include the rounds trained before the resume.
        """
        tree = read_checkpoint(node_dir, legacy=self.model.load_legacy)
        if tree is None:
            return False
        meta = tree.get("meta", {})
        self.round = int(meta.get("round", 0))
        self.rounds_accepted = int(meta.get("rounds_accepted", 0))
        self.best_round = int(meta.get("best_round", self.round))
        self.stall = int(meta.get("stall", 0))
        self.total_epochs = int(meta.get("total_epochs", 0))
        self.init = tree["init"]
        self.best = dict(tree["groups"])
        if not self.model.built:
            self.model.build(self.init)
        for name, state in self.best.items():
            self.model.import_state(name, state)
        if "proposal" in tree:
            self.model.import_proposal(tree["proposal"])
        return True

    def _save_history(self, path: Path) -> None:
        arrays = {}
        for key in ("epochs", "train_loss", "val_loss", "n_samples", "elapsed_min"):
            arrays[key] = np.asarray(self.history[key], dtype=float)
        for key in ("train_ids", "val_ids"):
            arrays[key] = np.asarray(self.history[key], dtype=float).reshape(-1, 2)
        for key, values in self.model.history.items():
            if values:
                arrays[f"model/{key}"] = np.stack([np.asarray(v) for v in values])
        arrays["rounds_json"] = np.array(json.dumps(self.history["rounds"]))
        np.savez(path, **arrays)
