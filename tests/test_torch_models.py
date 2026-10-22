"""Best state travelling from a trained model to a freshly built one (no Ray)."""

import numpy as np
import pytest
import torch

from falcon.core.model_sampler import ModelSampler
from falcon.core.raystore import Batch
from falcon.core.round_trainer import RoundTrainer
from falcon.core.state_io import load_state, save_state
from falcon.estimators import GaussianFullCov
from falcon.priors import Product

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
N_BINS = 20


def _batch(prior, n=128, seed=0):
    torch.manual_seed(seed)
    theta = torch.from_numpy(prior.simulate_batch(n))
    basis = torch.linspace(0, 1, N_BINS, dtype=torch.float64)
    x = theta[:, :1] + theta[:, 1:2] * basis + 0.1 * torch.randn(n, N_BINS, dtype=torch.float64)
    data = {"z.value": theta, "z.log_prob": torch.zeros(n, dtype=torch.float64), "x.value": x}
    return Batch(np.arange(n), data, None)


def _published(model, init):
    return {
        "meta": {"round": 1, "has_best": True},
        "init": init,
        "groups": {name: model.export_state(name) for name in model.groups()},
    }


def _gaussian(device):
    embedding = {
        "_target_": "falcon.embeddings.DynamicSVD",
        "_input_": ["x"],
        "n_components": 4,
        "buffer_size": 64,
        "whitener": {"_target_": "falcon.embeddings.DiagonalWhitener", "dim": N_BINS},
    }
    model = GaussianFullCov(embedding=embedding, device=device, gamma=0.5)
    model.setup(Product([["normal", 0.0, 1.0], ["normal", 0.0, 1.0]]), "z", ["x"])
    return model


def _trained_gaussian(device):
    model = _gaussian(device)
    batch = _batch(model.simulator_instance)
    init = model.init_from_batch(batch)
    model.build(init)
    for seed in range(3):
        model.train_step(_batch(model.simulator_instance, seed=seed))
    return model, init


def _assert_same_samples(model, sampler, mode="posterior"):
    x_obs = {"x": _batch(model.simulator_instance, n=1, seed=99)["x.value"].numpy()}
    expected = model.sample(np.random.default_rng(0), 50, x_obs, mode)
    actual = sampler.sample(mode, 50, x_obs, np.random.default_rng(0))
    np.testing.assert_allclose(actual["value"], expected["value"])
    np.testing.assert_allclose(actual["log_prob"], expected["log_prob"])


@pytest.mark.parametrize("device", DEVICES)
def test_gaussian_state_reaches_a_fresh_sampler(device):
    model, init = _trained_gaussian(device)
    assert model._model.embedding.modules_list[0].initialized  # the SVD basis was fitted

    sampler = ModelSampler(_gaussian(device))
    sampler.apply(_published(model, init))

    _assert_same_samples(model, sampler)
    _assert_same_samples(model, sampler, mode="proposal")
    batch = _batch(model.simulator_instance, seed=5)
    assert sampler.model.evaluate(batch) == pytest.approx(model.evaluate(batch))


def test_gaussian_state_survives_the_checkpoint_file(tmp_path):
    model, init = _trained_gaussian("cpu")
    save_state(tmp_path / "state.npz", _published(model, init))

    sampler = ModelSampler(_gaussian("cpu"))
    sampler.apply(load_state(tmp_path / "state.npz"))

    _assert_same_samples(model, sampler)


def test_gaussian_loads_legacy_checkpoint(tmp_path):
    model, init = _trained_gaussian("cpu")
    torch.save(model._model.state_dict(), tmp_path / "model.pth")
    torch.save(
        {"theta": torch.from_numpy(init["theta"]),
         "conditions": {k: torch.from_numpy(v) for k, v in init["conditions"].items()}},
        tmp_path / "init_tensors.pth",
    )
    torch.save({"rounds": 4, "accepted": 2, "stall": 1}, tmp_path / "round_state.pth")
    torch.save(7, tmp_path / "total_epochs_trained.pth")

    trainer = RoundTrainer(_gaussian("cpu"))
    assert trainer.load(tmp_path)
    assert (trainer.round, trainer.rounds_accepted, trainer.stall, trainer.total_epochs) == (4, 2, 1, 7)

    sampler = ModelSampler(_gaussian("cpu"))
    assert sampler.load(tmp_path)
    assert sampler.round == 4
    _assert_same_samples(model, sampler)


def _flow(device):
    from falcon.estimators import Flow

    model = Flow(device=device, num_proposals=32, net_type="zuko_nice")
    model.setup(Product([["uniform", -1.0, 1.0], ["uniform", -1.0, 1.0]]), "z", ["x"])
    return model


@pytest.mark.parametrize("device", DEVICES)
def test_flow_state_reaches_a_fresh_sampler(device, tmp_path):
    pytest.importorskip("sbi")
    model = _flow(device)
    init = model.init_from_batch(_batch(model.simulator_instance))
    model.build(init)
    for seed in range(3):
        model.train_step(_batch(model.simulator_instance, seed=seed))

    sampler = ModelSampler(_flow(device))
    sampler.apply(_published(model, init))
    _assert_same_samples(model, sampler)

    # Legacy files, as written before best_state.npz existed
    torch.save(model._conditional_flow.state_dict(), tmp_path / "conditional_flow.pth")
    torch.save(model._marginal_flow.state_dict(), tmp_path / "marginal_flow.pth")
    torch.save(model._embedding.state_dict(), tmp_path / "embedding.pth")
    torch.save([torch.from_numpy(init["theta"]),
                {k: torch.from_numpy(v) for k, v in init["conditions"].items()}],
               tmp_path / "init_parameters.pth")
    legacy = ModelSampler(_flow(device))
    assert legacy.load(tmp_path)
    _assert_same_samples(model, legacy)


def test_load_posterior_rebuilds_the_estimator_from_the_checkpoint(tmp_path):
    import falcon

    model, init = _trained_gaussian("cpu")
    tree = _published(model, init)
    tree["meta"].update(
        best_round=1,
        theta_key="z",
        condition_keys=["x"],
        artifact={
            "estimator": {"_target_": "falcon.estimators.GaussianFullCov", "gamma": 0.5,
                          "embedding": {
                              "_target_": "falcon.embeddings.DynamicSVD", "_input_": ["x"],
                              "n_components": 4, "buffer_size": 64,
                              "whitener": {"_target_": "falcon.embeddings.DiagonalWhitener", "dim": N_BINS},
                          }},
            "simulator": {"_target_": "falcon.priors.Product",
                          "priors": [["normal", 0.0, 1.0], ["normal", 0.0, 1.0]]},
        },
    )
    save_state(tmp_path / "best_state.npz", tree)

    post = falcon.load_posterior(tmp_path, device="cpu")

    assert post.node == "z" and post.condition_keys == ["x"] and post.round == 1
    _assert_same_samples(model, post._sampler)
    x_obs = {"x": _batch(model.simulator_instance, n=1, seed=99)["x.value"].numpy()}
    first = post.sample(10, x_obs, seed=3)["value"]
    np.testing.assert_array_equal(first, post.sample(10, x_obs, seed=3)["value"])


def test_load_posterior_needs_the_recorded_targets(tmp_path):
    import falcon

    model, init = _trained_gaussian("cpu")
    save_state(tmp_path / "best_state.npz", _published(model, init))
    with pytest.raises(ValueError, match="live"):
        falcon.load_posterior(tmp_path / "best_state.npz")
