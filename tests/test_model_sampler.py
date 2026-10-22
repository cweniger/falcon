"""ModelSampler, and the engines driving a model without any torch (no Ray)."""

import subprocess
import sys

import numpy as np
import pytest

from falcon.core.base_estimator import BaseEstimator
from falcon.core.model_sampler import ModelSampler
from falcon.core.round_trainer import RoundTrainer
from tests.fakes import FakeBuffer


class NumpyModel(BaseEstimator):
    """A model whose single "network" is one float, trained by counting steps."""

    def __init__(self, **overrides):
        self.max_epochs, self.patience_epochs, self.batch_size = 3, 1, 100
        self.max_rounds, self.prior_rounds = 2, 0
        for key, value in overrides.items():
            setattr(self, key, value)
        self.w = None
        self.setup(None, theta_key="theta", condition_keys=["x"])

    def groups(self):
        return {"main": "loss"}

    @property
    def built(self):
        return self.w is not None

    def init_from_batch(self, batch):
        return {"start": np.array(0.0)}

    def build(self, init_tree):
        self.w = float(init_tree["start"])

    def snapshot(self, group):
        return self.w

    def restore(self, group, handle):
        self.w = handle

    def export_state(self, group):
        return {"w": np.array(self.w)}

    def import_state(self, group, tree):
        self.w = float(tree["w"])

    def train_step(self, batch):
        self.w += 1.0
        return {"loss": 0.0}

    def evaluate(self, batch):
        return {"loss": -self.w}

    def discard_mask(self, batch):
        return np.zeros(len(batch), dtype=bool)

    def sample_prior(self, rng, num_samples):
        return {"value": rng.normal(size=(num_samples, 1)), "log_prob": np.zeros(num_samples)}

    def sample(self, rng, num_samples, conditions, mode):
        scale = 1.0 if mode == "posterior" else 0.5
        return {"value": np.full((num_samples, 1), self.w * scale), "log_prob": np.zeros(num_samples)}


def test_engines_drive_a_numpy_model_end_to_end():
    sampler = ModelSampler(NumpyModel())
    trainer = RoundTrainer(NumpyModel(), publish=sampler.apply)
    trainer.run(FakeBuffer())

    assert trainer.round == 2 and float(trainer.best["main"]["w"]) == 6.0
    assert sampler.status == "ready" and sampler.round == 2
    out = sampler.sample("posterior", 4, {"x": np.zeros((1, 1))}, np.random.default_rng(0))
    np.testing.assert_array_equal(out["value"], np.full((4, 1), 6.0))


def test_engines_do_not_import_torch():
    code = (
        "import sys\n"
        "import falcon.core.round_trainer, falcon.core.model_sampler, falcon.core.state_io\n"
        "assert 'torch' not in sys.modules, 'torch was imported'\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def _published(round_, w=3.0):
    return {"meta": {"round": round_}, "init": {"start": np.array(0.0)}, "groups": {"main": {"w": np.array(w)}}}


def test_prior_until_the_first_best_state():
    sampler = ModelSampler(NumpyModel())
    rng = np.random.default_rng(0)
    x = {"x": np.zeros((1, 1))}

    assert sampler.status == "waiting"
    for mode in ("proposal", "posterior"):
        assert sampler.uses_prior(mode)
    sampler.apply({"meta": {"round": 1}})  # counters only
    assert sampler.status == "waiting"

    sampler.apply(_published(1))
    assert sampler.status == "ready"
    np.testing.assert_array_equal(sampler.sample("proposal", 2, x, rng)["value"], [[1.5], [1.5]])
    assert sampler.samples_served == 2


def test_prior_proposals_during_prior_rounds():
    sampler = ModelSampler(NumpyModel(prior_rounds=2))
    sampler.apply(_published(2))

    assert sampler.status == "prior"
    assert sampler.uses_prior("proposal")
    assert not sampler.uses_prior("posterior")
    sampler.apply({"meta": {"round": 3}})
    assert not sampler.uses_prior("proposal")


def test_prior_sampling_rejects_conditions_and_bad_modes():
    sampler = ModelSampler(NumpyModel())
    with pytest.raises(ValueError):
        sampler.sample("prior", 2, {"x": np.zeros((1, 1))})
    with pytest.raises(ValueError):
        sampler.sample("likelihood", 2)


def test_weights_without_init_are_rejected():
    sampler = ModelSampler(NumpyModel())
    tree = _published(1)
    del tree["init"]
    with pytest.raises(RuntimeError, match="init"):
        sampler.apply(tree)


def test_load_from_checkpoint(tmp_path):
    trainer = RoundTrainer(NumpyModel())
    trainer.run(FakeBuffer())
    trainer.save(tmp_path)

    sampler = ModelSampler(NumpyModel())
    assert sampler.load(tmp_path)
    assert sampler.round == 2 and sampler.model.w == 6.0
    assert not ModelSampler(NumpyModel()).load(tmp_path / "missing")
