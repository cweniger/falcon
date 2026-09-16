"""Tests for the round-based training loop of RoundTrainer (no Ray).

The toy model's "networks" are scalar parameters that count training steps,
and validation losses are functions of those scalars. This makes the loop's
decisions (which epoch is the candidate, what gets promoted, where a round
starts) directly observable in the network state.
"""

from unittest.mock import MagicMock

import numpy as np
import torch
import torch.nn as nn

from falcon.core.round_trainer import RoundTrainer
from falcon.estimators.torch_model import NetworkGroup, TorchModel
from tests.fakes import FakeBuffer


class Scalar(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(()))

    def forward(self, conditions):
        return self.w


class ToyModel(TorchModel):
    """Primary group with an embedding, plus an auxiliary group without one."""

    def __init__(self, loss_fn, aux_fn=None, discard_ids=(), stop_at_step=None, **overrides):
        config = dict(
            max_epochs=10, patience_epochs=2, val_every_epochs=1, max_rounds=None,
            patience_rounds=2, prior_rounds=0, batch_size=100, max_cache_samples=0,
            cache_on_device=False, discard_samples=True, device="cpu",
        )
        config.update(overrides)
        for key, value in config.items():
            setattr(self, key, value)
        self.loss_fn = loss_fn
        self.aux_fn = aux_fn or (lambda w: 0.0)
        self.discard_ids = set(discard_ids)
        self.stop_at_step = stop_at_step
        self.trainer = None
        self.steps = 0
        self.events = []               # ("validate", round, epoch) / ("discard",) / ("publish", ...)
        self.validation_epochs = []    # (round, epoch) of in-round validations
        self.round_start_weights = []  # current primary weight at on_round_start
        self.discard_seen = []
        self.setup(MagicMock(param_dim=1), theta_key="theta", condition_keys=["x"])

    def init_from_batch(self, batch):
        return {}

    def build(self, init_tree):
        self.current, self.current_emb, self.current_aux = Scalar(), Scalar(), Scalar()
        self._set_groups({
            "primary": NetworkGroup({"net": self.current, "emb": self.current_emb}, "loss",
                                    embedding=self.current_emb),
            "aux": NetworkGroup({"net": self.current_aux}, "loss_aux"),
        })

    def train_step(self, batch):
        self._summary("primary", {}, train=True)
        with torch.no_grad():
            for net in (self.current, self.current_emb, self.current_aux):
                net.w += 1
        self.steps += 1
        if self.steps == self.stop_at_step:
            self.trainer.request_stop()
        return {"loss": 0.0}

    def val_step(self, batch):
        self.events.append(("validate", self.trainer.round, self.trainer.round_epoch))
        return {"loss": self.loss_fn(self.current.w.item()), "loss_aux": self.aux_fn(self.current_aux.w.item())}

    def on_validation_end(self, epoch, val_metrics):
        self.validation_epochs.append((self.trainer.round, epoch))
        return None

    def on_round_start(self):
        self.round_start_weights.append(self.current.w.item())

    def discard_test(self, batch):
        self.events.append(("discard",))
        self.discard_seen.extend(batch._ids.tolist())
        return np.isin(batch._ids, list(self.discard_ids))


def _trainer(model):
    published = []

    def publish(tree):
        published.append(tree)
        model.events.append(("publish", "groups" in tree))

    trainer = RoundTrainer(model, publish=publish)
    trainer.published = published
    model.trainer = trainer
    return trainer


def _train(model, buffer=None):
    buffer = buffer or FakeBuffer()
    trainer = _trainer(model)
    trainer.run(buffer)
    return trainer, buffer


def _best_w(trainer, group="primary", module="net"):
    return float(trainer.best[group][module]["w"])


def _parabola(w):
    """Minimum at w=2: training past two epochs makes the network worse."""
    return (w - 2.0) ** 2 + 1.0


def test_candidate_is_best_validated_epoch():
    model = ToyModel(_parabola, max_rounds=1)
    trainer, _ = _train(model)

    # Losses per epoch: 2, 1, 2, 5 -> round ends at epoch 4, two epochs after the best
    assert model.validation_epochs == [(1, 1), (1, 2), (1, 3), (1, 4)]
    assert _best_w(trainer) == 2.0
    assert _best_w(trainer, module="emb") == 2.0  # embedding travels with its density network
    record = trainer.history["rounds"][0]
    assert record["accepted"]
    assert record["groups"]["primary"]["best"] is None  # first round: nothing to compare against
    assert record["groups"]["primary"]["candidate"] == 1.0
    assert record["epochs"] == 4


def test_rejected_rounds_restart_from_best_and_stop_at_patience_rounds():
    model = ToyModel(_parabola, patience_rounds=2)
    trainer, buffer = _train(model)

    assert [r["accepted"] for r in trainer.history["rounds"]] == [True, False, False]
    assert trainer.round == 3 and trainer.stall == 2 and trainer.rounds_accepted == 1
    # Rounds 2 and 3 both start from the best network (w=2), not from where round 2 ended
    assert model.round_start_weights == [2.0, 2.0]
    assert _best_w(trainer) == 2.0
    assert buffer.train.refreshes == 3 + 1  # once per round, plus before the one sweep


def test_best_network_is_validated_before_the_first_epoch():
    model = ToyModel(_parabola, patience_rounds=1)
    trainer, _ = _train(model)

    validations = [e[1:] for e in model.events if e[0] == "validate"]
    assert validations[0] == (1, 1)                 # round 1 has no best network yet
    assert (2, 0) in validations                    # round 2 starts with the best network
    assert validations.index((2, 0)) < validations.index((2, 1))
    record = trainer.history["rounds"][1]
    assert record["groups"]["primary"]["best"] == _parabola(2.0)


def test_publishes_counters_every_round_and_weights_after_promotions():
    model = ToyModel(_parabola, patience_rounds=2)
    trainer, _ = _train(model)

    assert [("groups" in t, "init" in t) for t in trainer.published] == [
        (False, False),  # round 1 starts
        (True, True),    # round 1 promoted both groups; init is sent once
        (False, False),  # round 2 starts; nothing promoted
        (False, False),  # round 3 starts; nothing promoted
    ]
    assert [t["meta"]["round"] for t in trainer.published] == [1, 1, 2, 3]
    groups = trainer.published[1]["groups"]
    assert float(groups["primary"]["net"]["w"]) == 2.0
    assert trainer.published[1]["meta"]["has_best"]


def test_publishes_weights_when_only_an_auxiliary_group_improves():
    model = ToyModel(_parabola, aux_fn=lambda w: -w, max_rounds=2)
    trainer, _ = _train(model)

    assert not trainer.history["rounds"][1]["accepted"]
    full = [t for t in trainer.published if "groups" in t]
    assert len(full) == 2
    assert "init" not in full[1]
    assert float(full[1]["groups"]["aux"]["net"]["w"]) == 7.0


def test_discard_sweep_runs_after_publish_and_covers_train_and_val():
    model = ToyModel(_parabola, patience_rounds=2, discard_ids={1, 3, 101})
    trainer, buffer = _train(model)

    buffer.dataset_manager.deactivate.remote.assert_called_once()
    (ids,), _ = buffer.dataset_manager.deactivate.remote.call_args
    assert sorted(ids) == [1, 3, 101]
    assert sorted(model.discard_seen) == list(range(10)) + list(range(100, 105))
    assert trainer.history["rounds"][0]["n_discarded_train"] == 2
    assert trainer.history["rounds"][0]["n_discarded_val"] == 1
    first_discard = model.events.index(("discard",))
    assert model.events.index(("publish", True)) < first_discard


def test_no_discard_sweep_when_disabled():
    model = ToyModel(_parabola, max_rounds=1, discard_samples=False, discard_ids={1})
    trainer, buffer = _train(model)

    buffer.dataset_manager.deactivate.remote.assert_not_called()
    assert model.discard_seen == []


def test_groups_are_promoted_independently():
    # Primary gets worse after round 1; aux keeps improving with more training
    model = ToyModel(_parabola, aux_fn=lambda w: -w, max_rounds=2)
    trainer, _ = _train(model)

    round2 = trainer.history["rounds"][1]
    assert not round2["accepted"]
    assert not round2["groups"]["primary"]["promoted"]
    assert round2["groups"]["aux"]["promoted"]
    assert _best_w(trainer) == 2.0
    # Round 1 ends at epoch 4 (aux w=4); round 2 starts there and stops after 3 epochs
    assert _best_w(trainer, group="aux") == 7.0


def test_stops_at_max_rounds():
    model = ToyModel(lambda w: -w, max_rounds=3, max_epochs=2)
    trainer, _ = _train(model)

    assert trainer.round == 3 and trainer.rounds_accepted == 3


def test_validation_interval_and_patience_counted_in_epochs():
    model = ToyModel(lambda w: 1.0, max_rounds=1, max_epochs=20, val_every_epochs=3, patience_epochs=6)
    _train(model)

    # Best at epoch 3, never improves; 9 - 3 >= 6 ends the round at the third validation
    assert model.validation_epochs == [(1, 3), (1, 6), (1, 9)]


def test_last_epoch_is_always_validated():
    model = ToyModel(lambda w: -w, max_rounds=1, max_epochs=6, val_every_epochs=4, patience_epochs=100)
    trainer, _ = _train(model)

    assert model.validation_epochs == [(1, 4), (1, 6)]
    assert _best_w(trainer) == 6.0


def test_epochs_to_validations():
    trainer = RoundTrainer(ToyModel(_parabola, val_every_epochs=3))
    assert [trainer._epochs_to_validations(e) for e in (0, 1, 3, 8, 9)] == [0, 1, 1, 3, 3]


def test_stop_request_ends_within_a_step_and_still_runs_acceptance():
    # 10 training samples, batch size 2 -> 5 steps per epoch; stop in the middle of epoch 2
    model = ToyModel(lambda w: -w, batch_size=2, stop_at_step=7, discard_ids={1})
    trainer, buffer = _train(model)

    assert model.steps == 7
    assert model.validation_epochs == [(1, 1)]
    # The partial epoch is dropped: the candidate is the state validated after epoch 1
    assert _best_w(trainer) == 5.0
    assert trainer.has_best and trainer.rounds_accepted == 1
    assert "groups" in trainer.published[-1]  # the final best state still reaches the samplers
    buffer.dataset_manager.deactivate.remote.assert_not_called()


def test_stop_request_before_first_validation_tests_nothing():
    model = ToyModel(lambda w: -w, batch_size=2, stop_at_step=3)
    trainer, _ = _train(model)

    assert model.steps == 3
    assert not trainer.has_best
    assert trainer.history["rounds"] == []


def test_stop_request_during_round_start_refresh_starts_no_round():
    model = ToyModel(_parabola, discard_samples=False)
    buffer = FakeBuffer()
    trainer = _trainer(model)

    def refresh_then_stop():
        buffer.train.refreshes += 1
        if buffer.train.refreshes == 2:  # the refresh at the start of round 2
            trainer.request_stop()

    buffer.train.refresh = refresh_then_stop
    trainer.run(buffer)

    assert trainer.round == 1
    assert model.steps == 4  # round 1 only: no step taken after the stop


def test_stop_request_during_discard_sweep_skips_remaining_samples():
    model = ToyModel(_parabola, max_rounds=1, batch_size=4, discard_ids={1, 101})
    original_test = model.discard_test

    def test_then_stop(batch):
        model.trainer.request_stop()
        return original_test(batch)

    model.discard_test = test_then_stop
    _, buffer = _train(model)

    assert model.discard_seen == [0, 1, 2, 3]  # first training batch only, no validation batch
    (ids,), _ = buffer.dataset_manager.deactivate.remote.call_args
    assert ids == [1]


def test_save_and_resume(tmp_path):
    model = ToyModel(_parabola, patience_rounds=2)
    trainer, _ = _train(model)
    assert trainer.save(tmp_path)

    resumed_model = ToyModel(_parabola)
    resumed = RoundTrainer(resumed_model)
    assert resumed.load(tmp_path)

    assert (resumed.round, resumed.rounds_accepted, resumed.stall) == (3, 1, 2)
    assert resumed.has_best
    assert _best_w(resumed) == 2.0
    assert resumed_model.current.w.item() == 2.0  # the current networks start from the best
    assert (tmp_path / "training_history.npz").exists()


def test_save_without_a_completed_round_writes_nothing(tmp_path):
    trainer = RoundTrainer(ToyModel(_parabola))
    assert not trainer.save(tmp_path)
    assert not any(tmp_path.iterdir())
