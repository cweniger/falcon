"""Tests for the round-based training loop of StepwiseEstimator (no Ray).

The toy estimator's "networks" are scalar parameters that count training
steps, and validation losses are functions of those scalars. This makes the
loop's decisions (which epoch is the candidate, what gets promoted, where a
round starts) directly observable in the network state.
"""

import asyncio
from unittest.mock import MagicMock

import numpy as np
import torch
import torch.nn as nn

from falcon.core.raystore import Batch
from falcon.estimators.stepwise_base import NetworkGroup, StepwiseEstimator


class Scalar(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(()))

    def forward(self, conditions):
        return self.w


class ToyEstimator(StepwiseEstimator):
    """Primary group with an embedding, plus an auxiliary group without one."""

    def __init__(self, loss_fn, aux_fn=None, discard_ids=(), terminate_at_step=None, **overrides):
        config = dict(
            max_epochs=10, patience_epochs=2, val_every_epochs=1, max_rounds=None,
            patience_rounds=2, prior_rounds=0, batch_size=100, max_cache_samples=0,
            cache_on_device=False, discard_samples=True,
        )
        config.update(overrides)
        for key, value in config.items():
            setattr(self, key, value)
        self.loss_fn = loss_fn
        self.aux_fn = aux_fn or (lambda w: 0.0)
        self.discard_ids = set(discard_ids)
        self.terminate_at_step = terminate_at_step
        self.steps = 0
        self.validation_epochs = []   # (round, epoch) of in-round validations
        self.round_start_weights = []  # current primary weight at on_round_start
        self.discard_seen = []
        self.setup(MagicMock(param_dim=1), theta_key="theta", condition_keys=["x"])

    def _init_networks(self):
        self.current, self.best = Scalar(), Scalar()
        self.current_emb, self.best_emb = Scalar(), Scalar()
        self.current_aux, self.best_aux = Scalar(), Scalar()
        self.networks_initialized = True

    def _network_groups(self):
        return {
            "primary": NetworkGroup(self.current, self.best, "loss", self.current_emb, self.best_emb),
            "aux": NetworkGroup(self.current_aux, self.best_aux, "loss_aux"),
        }

    def train_step(self, batch):
        if not self.networks_initialized:
            self._init_networks()
        self._summary(batch, "primary", {}, train=True)
        with torch.no_grad():
            for net in (self.current, self.current_emb, self.current_aux):
                net.w += 1
        self.steps += 1
        if self.steps == self.terminate_at_step:
            self.interrupt()
        return {"loss": 0.0}

    def val_step(self, batch, use_best=False):
        groups = self._network_groups()
        primary = groups["primary"].best if use_best else groups["primary"].current
        aux = groups["aux"].best if use_best else groups["aux"].current
        return {"loss": self.loss_fn(primary.w.item()), "loss_aux": self.aux_fn(aux.w.item())}

    def on_validation_end(self, epoch, val_metrics):
        self.validation_epochs.append((self._round, epoch))
        return None

    def on_round_start(self):
        self.round_start_weights.append(self.current.w.item())

    def discard_mask(self, batch):
        self.discard_seen.extend(batch._ids.tolist())
        return np.isin(batch._ids, list(self.discard_ids))

    def sample_prior(self, num_samples, conditions=None):
        raise NotImplementedError

    def sample_posterior(self, num_samples, conditions=None):
        raise NotImplementedError

    def sample_proposal(self, num_samples, conditions=None):
        raise NotImplementedError

    def save(self, node_dir):
        raise NotImplementedError

    def load(self, node_dir):
        raise NotImplementedError


class FakeLoader:
    def __init__(self, ids, dataset_manager):
        self.ids = np.asarray(ids)
        self.dataset_manager = dataset_manager
        self.refreshes = 0

    @property
    def count(self):
        return len(self.ids)

    async def refresh(self):
        self.refreshes += 1

    def iter_batches(self, batch_size, shuffle=False, drop_last=False):
        for start in range(0, len(self.ids), batch_size):
            yield Batch(self.ids[start:start + batch_size], {}, self.dataset_manager)


class FakeBuffer:
    def __init__(self):
        self.dataset_manager = MagicMock()
        self.train = FakeLoader(range(10), self.dataset_manager)
        self.val = FakeLoader(range(100, 105), self.dataset_manager)

    def cached_loader(self, keys, max_cache_samples=0):
        return self.train

    def cached_val_loader(self, keys, max_cache_samples=0):
        return self.val

    def get_stats(self):
        return {"total_length": 15}


def _train(estimator):
    buffer = FakeBuffer()
    asyncio.run(estimator.train(buffer))
    return buffer


def _parabola(w):
    """Minimum at w=2: training past two epochs makes the network worse."""
    return (w - 2.0) ** 2 + 1.0


def test_candidate_is_best_validated_epoch():
    est = ToyEstimator(_parabola, max_rounds=1)
    _train(est)

    # Losses per epoch: 2, 1, 2, 5 -> round ends at epoch 4, two epochs after the best
    assert est.validation_epochs == [(1, 1), (1, 2), (1, 3), (1, 4)]
    assert est.best.w.item() == 2.0
    assert est.best_emb.w.item() == 2.0  # embedding travels with its density network
    record = est.history["rounds"][0]
    assert record["accepted"]
    assert record["groups"]["primary"]["best"] is None  # first round: nothing to compare against
    assert record["groups"]["primary"]["candidate"] == 1.0
    assert record["epochs"] == 4


def test_rejected_rounds_restart_from_best_and_stop_at_patience_rounds():
    est = ToyEstimator(_parabola, patience_rounds=2)
    buffer = _train(est)

    assert [r["accepted"] for r in est.history["rounds"]] == [True, False, False]
    assert est._round == 3 and est._stall == 2 and est._rounds_accepted == 1
    # Rounds 2 and 3 both start from the best network (w=2), not from where round 2 ended
    assert est.round_start_weights == [2.0, 2.0]
    assert est.best.w.item() == 2.0
    assert buffer.train.refreshes == 3 + 1  # once per round, plus before the one sweep


def test_discard_sweep_only_after_accepted_rounds_covers_train_and_val():
    est = ToyEstimator(_parabola, patience_rounds=2, discard_ids={1, 3, 101})
    buffer = _train(est)

    buffer.dataset_manager.deactivate.remote.assert_called_once()
    (ids,), _ = buffer.dataset_manager.deactivate.remote.call_args
    assert sorted(ids) == [1, 3, 101]
    assert sorted(est.discard_seen) == list(range(10)) + list(range(100, 105))
    assert est.history["rounds"][0]["n_discarded_train"] == 2
    assert est.history["rounds"][0]["n_discarded_val"] == 1


def test_no_discard_sweep_when_disabled():
    est = ToyEstimator(_parabola, max_rounds=1, discard_samples=False, discard_ids={1})
    buffer = _train(est)

    buffer.dataset_manager.deactivate.remote.assert_not_called()
    assert est.discard_seen == []


def test_groups_are_promoted_independently():
    # Primary gets worse after round 1; aux keeps improving with more training
    est = ToyEstimator(_parabola, aux_fn=lambda w: -w, max_rounds=2)
    _train(est)

    round2 = est.history["rounds"][1]
    assert not round2["accepted"]
    assert not round2["groups"]["primary"]["promoted"]
    assert round2["groups"]["aux"]["promoted"]
    assert est.best.w.item() == 2.0
    # Round 1 ends at epoch 4 (aux w=4); round 2 starts there and stops after 3 epochs
    assert est.best_aux.w.item() == 7.0


def test_stops_at_max_rounds():
    est = ToyEstimator(lambda w: -w, max_rounds=3, max_epochs=2)
    _train(est)

    assert est._round == 3 and est._rounds_accepted == 3


def test_validation_interval_and_patience_counted_in_epochs():
    est = ToyEstimator(lambda w: 1.0, max_rounds=1, max_epochs=20, val_every_epochs=3, patience_epochs=6)
    _train(est)

    # Best at epoch 3, never improves; 9 - 3 >= 6 ends the round at the third validation
    assert est.validation_epochs == [(1, 3), (1, 6), (1, 9)]


def test_last_epoch_is_always_validated():
    est = ToyEstimator(lambda w: -w, max_rounds=1, max_epochs=6, val_every_epochs=4, patience_epochs=100)
    _train(est)

    assert est.validation_epochs == [(1, 4), (1, 6)]
    assert est.best.w.item() == 6.0


def test_epochs_to_validations():
    est = ToyEstimator(_parabola, val_every_epochs=3)
    assert [est._epochs_to_validations(e) for e in (0, 1, 3, 8, 9)] == [0, 1, 1, 3, 3]


def test_interrupt_stops_within_a_step_and_still_runs_acceptance():
    # 10 training samples, batch size 2 -> 5 steps per epoch; stop in the middle of epoch 2
    est = ToyEstimator(lambda w: -w, batch_size=2, terminate_at_step=7, discard_ids={1})
    buffer = _train(est)

    assert est.steps == 7
    assert est.validation_epochs == [(1, 1)]
    # The partial epoch is dropped: the candidate is the state validated after epoch 1
    assert est.best.w.item() == 5.0
    assert est._has_best and est._rounds_accepted == 1
    buffer.dataset_manager.deactivate.remote.assert_not_called()


def test_interrupt_before_first_validation_tests_nothing():
    est = ToyEstimator(lambda w: -w, batch_size=2, terminate_at_step=3)
    _train(est)

    assert est.steps == 3
    assert not est._has_best
    assert est.history["rounds"] == []


def test_interrupt_during_round_start_refresh_starts_no_round():
    est = ToyEstimator(_parabola, discard_samples=False)
    buffer = FakeBuffer()

    async def refresh_then_stop():
        buffer.train.refreshes += 1
        if buffer.train.refreshes == 2:  # the refresh at the start of round 2
            est.interrupt()

    buffer.train.refresh = refresh_then_stop
    asyncio.run(est.train(buffer))

    assert est._round == 1
    assert est.steps == 4  # round 1 only: no step taken after the stop


def test_interrupt_during_discard_sweep_skips_remaining_samples():
    est = ToyEstimator(_parabola, max_rounds=1, batch_size=4, discard_ids={1, 101})
    original_mask = est.discard_mask

    def mask_then_stop(batch):
        est.interrupt()
        return original_mask(batch)

    est.discard_mask = mask_then_stop
    buffer = _train(est)

    assert est.discard_seen == [0, 1, 2, 3]  # first training batch only, no validation batch
    (ids,), _ = buffer.dataset_manager.deactivate.remote.call_args
    assert ids == [1]


def test_round_state_round_trip(tmp_path):
    est = ToyEstimator(_parabola)
    est._round, est._rounds_accepted, est._stall = 5, 3, 2
    est._save_round_state(tmp_path)

    resumed = ToyEstimator(_parabola)
    resumed._load_round_state(tmp_path)
    assert (resumed._round, resumed._rounds_accepted, resumed._stall) == (5, 3, 2)
    assert resumed._has_best


def test_prior_proposal_until_best_network_and_prior_rounds():
    est = ToyEstimator(lambda w: -w, prior_rounds=2)
    assert est._use_prior_proposal()          # no best network yet
    est._has_best, est._round = True, 2
    assert est._use_prior_proposal()          # still within prior_rounds
    est._round = 3
    assert not est._use_prior_proposal()
