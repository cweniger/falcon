"""FlowMatching: flow-matching building blocks, the estimator, and the region ladder (no Ray)."""

import math
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from falcon.core.model_sampler import ModelSampler
from falcon.core.raystore import Batch
from falcon.core.state_io import load_state, save_state
from falcon.estimators import FlowMatching
from falcon.estimators.flow_matching import VelocityField, Whitener, cnf_logprob, copy_buffers, euler_sample
from falcon.estimators.region_ladder import PRIOR, FlowPair, LadderConfig, RegionLadder, leak, log_normal
from falcon.priors import Product

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
SIGMA = 0.1


# ==================== Building blocks ====================


def _zero_field(param_dim=2, cond_dim=3):
    net = VelocityField(param_dim, cond_dim, hidden=8, layers=2, time_dim=4)
    with torch.no_grad():
        net.net[-1].weight.zero_()
        net.net[-1].bias.zero_()
    return net


def test_zero_velocity_field_is_the_base_distribution():
    torch.manual_seed(0)
    net = _zero_field()
    w = torch.randn(1000, 2)
    s = torch.randn(1000, 3)
    lp = cnf_logprob(net, w, s, steps=4)
    expected = -0.5 * (w.pow(2).sum(1) + 2 * math.log(2 * math.pi))
    torch.testing.assert_close(lp, expected)
    draws = euler_sample(net, torch.zeros(20000, 3), steps=4)
    assert draws.std(0).sub(1).abs().max() < 0.03


def test_hutchinson_divergence_matches_exact_on_a_linear_field():
    # v(w) = A w has divergence tr(A) everywhere; Rademacher probes estimate it
    class Linear(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.A = torch.tensor([[0.3, 0.1], [-0.2, -0.5]])

        def forward(self, w, t, s):
            return w @ self.A.T

    torch.manual_seed(1)
    w = torch.randn(4000, 2)
    exact = cnf_logprob(Linear(), w, torch.zeros(4000, 1), steps=8, divergence="exact")
    hutch = cnf_logprob(Linear(), w, torch.zeros(4000, 1), steps=8, divergence="hutchinson", n_probe=4)
    assert (exact - hutch).abs().mean() < 0.1


def test_whitener_is_an_exact_zca_refit():
    torch.manual_seed(0)
    L = torch.tensor([[2.0, 0.0], [1.5, 0.5]], dtype=torch.float64)
    u = torch.randn(50000, 2, dtype=torch.float64) @ L.T + torch.tensor([1.0, -3.0], dtype=torch.float64)
    whitener = Whitener(2)
    whitener.refit(u)
    w = whitener.whiten(u)
    torch.testing.assert_close(torch.cov(w.T), torch.eye(2, dtype=torch.float64), atol=1e-8, rtol=0)
    torch.testing.assert_close(whitener.unwhiten(w), u)
    cov = torch.cov(u.T)
    assert float(whitener.logdet) == pytest.approx(-0.5 * float(torch.logdet(cov)))
    torch.testing.assert_close(whitener.W, whitener.W.T)  # symmetric root: stays in the latent frame


def test_copy_buffers_creates_lazily_built_buffers():
    class Lazy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("stats", None)

    src, dst = Lazy(), Lazy()
    src.stats = torch.ones(3)
    copy_buffers(dst, src)
    torch.testing.assert_close(dst.stats, torch.ones(3))
    assert "stats" in dst.state_dict()


# ==================== Estimator ====================


def _prior():
    return Product([["normal", 0.0, 1.0], ["normal", 0.0, 1.0]])


def _model(device="cpu", embedding=None, **overrides):
    config = dict(
        hidden=16, layers=2, time_dim=4, sample_steps=4, density_steps=4,
        n_region=256, n_mout=256, v_min_ess=10, v_max_draws=1024, readout_draws=256, device=device,
    )
    config.update(overrides)
    model = FlowMatching(embedding=embedding, **config)
    model.setup(_prior(), "z", ["x"])
    return model


def _batch(n=128, seed=0, shift=0.0, scale=1.0, dtype=torch.float64):
    g = torch.Generator().manual_seed(seed)
    theta = torch.randn(n, 2, generator=g, dtype=torch.float64) * scale + shift
    x = (theta + SIGMA * torch.randn(n, 2, generator=g, dtype=torch.float64)).to(dtype)
    return Batch(np.arange(n), {"z.value": theta, "z.log_prob": torch.zeros(n, dtype=torch.float64),
                                "x.value": x}, None)


def _trained(device="cpu", **overrides):
    model = _model(device, **overrides)
    batches = [_batch(seed=s) for s in range(4)]
    init = model.init_from_batch(batches[0])
    model.build(init)
    model.on_train_start(iter(batches))
    for batch in batches:
        model.train_step(batch)
    return model, init


def _published(model, init, full=False):
    return {
        "meta": {"round": 1, "has_best": True},
        "init": init,
        "groups": {name: model.export_state(name) for name in model.groups()},
        "proposal": model.export_proposal(full=full),
    }


X_OBS = {"x": np.array([[0.5, -0.3]])}


def _assert_same_samples(model, sampler, mode):
    expected = model.sample(np.random.default_rng(0), 50, X_OBS, mode)
    actual = sampler.sample(mode, 50, X_OBS, np.random.default_rng(0))
    np.testing.assert_allclose(actual["value"], expected["value"])
    np.testing.assert_allclose(actual["log_prob"], expected["log_prob"])


def test_requires_a_transformed_prior():
    with pytest.raises(TypeError, match="TransformedPrior"):
        FlowMatching().setup(MagicMock(), "z", ["x"])


def test_groups_and_validation_metrics():
    model, _ = _trained()
    assert model.groups() == {"conditional": "nll", "marginal": "nll_aux"}
    metrics = model.evaluate(_batch(seed=9))
    assert set(metrics) == {"loss", "nll", "nll_aux", "fm", "fm_aux"}
    assert all(np.isfinite(v) for v in metrics.values())


@pytest.mark.parametrize("device", DEVICES)
def test_state_reaches_a_fresh_sampler(device, tmp_path):
    model, init = _trained(device)
    model.set_observations(X_OBS)
    model.on_round_end({"conditional": True, "marginal": True})

    sampler = ModelSampler(_model(device))
    sampler.apply(_published(model, init))
    _assert_same_samples(model, sampler, "posterior")
    _assert_same_samples(model, sampler, "proposal")

    # The checkpoint format: arrays and plain values only
    save_state(tmp_path / "state.npz", _published(model, init, full=True))
    from_file = ModelSampler(_model(device))
    from_file.apply(load_state(tmp_path / "state.npz"))
    _assert_same_samples(model, from_file, "posterior")


def test_whitener_refit_holds_the_frame_of_a_flow_that_was_not_promoted():
    model, _ = _trained()
    mean_c = model._flow_c.whitener.mean.clone()
    model.on_round_end({"conditional": False, "marginal": True})  # no observation: the proposal stays the prior
    model.on_round_start()
    model.on_train_start(iter([_batch(seed=s, shift=2.0, scale=0.5) for s in range(4)]))
    torch.testing.assert_close(model._flow_c.whitener.mean, mean_c)
    assert model._flow_m.whitener.mean.sub(2.0).abs().max() < 0.1
    assert float(model._flow_m.whitener.logdet) == pytest.approx(-2 * math.log(0.5), abs=0.2)


def test_embedding_trains_only_in_the_first_embedding_epochs():
    embedding = {"_target_": "torch.nn.Linear", "_input_": ["x"], "in_features": 2, "out_features": 3}
    model = _model(embedding=embedding, embedding_epochs=1)
    batches = [_batch(seed=s, dtype=torch.float32) for s in range(4)]  # 512 samples: 4 steps per epoch
    model.build(model.init_from_batch(batches[0]))
    model.on_train_start(iter(batches))
    trained = model._train_nets["embedding"]
    for batch in batches:
        model.train_step(batch)
    after_first_epoch = {k: v.clone() for k, v in trained.state_dict().items()}
    ema_after_first_epoch = {k: v.clone() for k, v in model._embedding.state_dict().items()}
    for batch in batches:
        model.train_step(batch)
    for key, value in trained.state_dict().items():
        torch.testing.assert_close(value, after_first_epoch[key])
    for key, value in model._embedding.state_dict().items():
        torch.testing.assert_close(value, ema_after_first_epoch[key])


def test_posterior_needs_a_single_observation():
    model, _ = _trained()
    with pytest.raises(ValueError, match="one observation"):
        model.sample(np.random.default_rng(0), 5, {"x": np.zeros((2, 2))}, "posterior")


def test_without_observation_the_proposal_is_the_prior():
    model, _ = _trained()
    assert not model.on_round_end({"conditional": True, "marginal": True})
    draws = model.sample(np.random.default_rng(0), 2000, X_OBS, "proposal")
    assert np.abs(draws["value"].std(0) - 1).max() < 0.1
    assert not model.discard_mask(_batch(seed=3)).any()


# ==================== Region ladder ====================


class GaussianFlow:
    """Isotropic N(0, sigma^2 I) in two dimensions, with the flow interface."""

    def __init__(self, sigma):
        self.sigma = float(sigma)

    def sample(self, s):
        return self.sigma * torch.randn(len(s), 2, dtype=torch.float64)

    def log_prob(self, u, s):
        return -0.5 * u.pow(2).sum(1) / self.sigma ** 2 - math.log(2 * math.pi * self.sigma ** 2)


def _pair(sigma_c, sigma_m=1.0):
    return FlowPair(GaussianFlow(sigma_c), GaussianFlow(sigma_m), torch.zeros(1, 1))


def _trees(sigma_c, sigma_m=1.0):
    return {"c": {"sigma": np.array(sigma_c)}, "m": {"sigma": np.array(sigma_m)}, "s_obs": np.zeros((1, 1))}


def _ladder():
    config = LadderConfig(n_region=20000, n_mout=20000)
    return RegionLadder(config, lambda r: _pair(float(r["c"]["sigma"]), float(r["m"]["sigma"])), 2, "cpu")


def _radius2(region, sigma):
    """Squared radius of the level set ln N(u; 0, sigma^2 I) > thr."""
    return -2 * sigma ** 2 * (region["thr"] + math.log(2 * math.pi * sigma ** 2))


def test_first_step_contracts_to_the_level_set_of_the_conditional_flow():
    torch.manual_seed(0)
    ladder = _ladder()
    assert ladder.step(_pair(0.3), _trees(0.3)) == "CONTRACT"
    assert (ladder.inner, ladder.outer, ladder.stack) == ("r001", PRIOR, [PRIOR])

    region = ladder.regions["r001"]
    c = -2 * math.log(leak(4.0))  # chi^2_2 quantile holding all but leak(4) of the mass
    assert region["thr"] == pytest.approx(-c / 2 - math.log(2 * math.pi * 0.09), abs=0.3)
    assert region["V"] == pytest.approx(1 - math.exp(-_radius2(region, 0.3) / 2), abs=0.01)
    assert region["V_cut"] == pytest.approx(region["V"])


def _contracted_twice():
    torch.manual_seed(0)
    ladder = _ladder()
    ladder.step(_pair(0.3), _trees(0.3))
    assert ladder.step(_pair(0.1), _trees(0.1)) == "CONTRACT"
    return ladder


def test_proposal_samples_the_prior_truncated_to_the_outer_region():
    ladder = _contracted_twice()
    assert (ladder.inner, ladder.outer, ladder.stack) == ("r002", "r001", [PRIOR, PRIOR])

    u, log_prob = ladder.sample(3000)
    r2 = _radius2(ladder.regions["r001"], 0.3)
    assert (u.pow(2).sum(1) < r2).all()
    assert len(torch.unique(u, dim=0)) == 3000
    torch.testing.assert_close(log_prob, log_normal(u) - math.log(ladder.volume("r001")))
    # Uniform in the prior on the disc: the radius^2 follows the truncated chi^2_2
    expected = 2 - r2 * math.exp(-r2 / 2) / (1 - math.exp(-r2 / 2))
    assert float(u.pow(2).sum(1).mean()) == pytest.approx(expected, rel=0.05)


def test_hold_when_the_candidate_is_not_a_real_step_down():
    ladder = _contracted_twice()
    action = ladder.step(_pair(0.1), _trees(0.1))
    assert action.startswith("hold (R")
    assert (ladder.inner, ladder.outer) == ("r002", "r001")


def test_expand_pops_the_stack_and_forgets_the_dropped_region():
    ladder = _contracted_twice()
    assert ladder.step(_pair(1.0), _trees(1.0)) == "EXPAND"
    assert (ladder.inner, ladder.outer, ladder.stack) == ("r001", PRIOR, [PRIOR])
    assert set(ladder.regions) == {"r001"}


def test_ladder_state_survives_export_and_the_checkpoint_file(tmp_path):
    ladder = _contracted_twice()
    live = ladder.export(full=False)
    assert set(live["regions"]) == {"r001", "r002"} and live["stack"] == []

    save_state(tmp_path / "ladder.npz", {"proposal": ladder.export(full=True)})
    restored = _ladder()
    restored.load(load_state(tmp_path / "ladder.npz")["proposal"])
    assert (restored.inner, restored.outer, restored.stack) == (ladder.inner, ladder.outer, ladder.stack)
    u = torch.randn(1000, 2, dtype=torch.float64)
    torch.testing.assert_close(restored.cut("r002", u), ladder.cut("r002", u))
