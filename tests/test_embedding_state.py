"""Embedding modules whose buffers are created from the first data (no Ray)."""

import pytest
import torch

from falcon.embeddings import DiagonalWhitener, DynamicSVD, RunningNorm, ToeplitzWhitener

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _data(n=64, d=20, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, d, generator=g) * 3 + 1


def _fitted_svd():
    torch.manual_seed(0)
    svd = DynamicSVD(n_components=4, buffer_size=32, whitener=DiagonalWhitener(dim=20))
    svd.train()
    for seed in range(4):
        x = _data(seed=seed)
        svd(x, x - 0.1 * _data(seed=seed + 10))
    svd.eval()
    return svd


def test_running_norm_state_loads_into_fresh_module():
    norm = RunningNorm(momentum=0.1)
    norm.train()
    for seed in range(3):
        norm(_data(seed=seed))
    norm.eval()

    fresh = RunningNorm(momentum=0.1)
    assert not fresh.initialized
    fresh.load_state_dict(norm.state_dict())
    fresh.eval()

    assert fresh.initialized
    x = _data(seed=5)
    torch.testing.assert_close(fresh(x), norm(x))


def test_toeplitz_whitener_state_loads_into_fresh_module():
    whitener = ToeplitzWhitener()
    whitener.update(_data(seed=1))

    fresh = ToeplitzWhitener()
    fresh.load_state_dict(whitener.state_dict())

    assert fresh.initialized
    x = _data(seed=2)
    torch.testing.assert_close(fresh(x), whitener(x))


def test_loading_a_state_without_the_buffer_resets_it():
    updated = ToeplitzWhitener()
    updated.update(_data())

    updated.load_state_dict(ToeplitzWhitener().state_dict())

    assert not updated.initialized
    assert updated.running_var is None


def test_diagonal_whitener_keeps_loaded_statistics_on_the_next_update():
    whitener = DiagonalWhitener(dim=20, momentum=0.5)
    whitener.update(_data(seed=1))

    resumed = DiagonalWhitener(dim=20, momentum=0.5)
    resumed.load_state_dict(whitener.state_dict())
    batch = _data(seed=2)
    resumed.update(batch)

    expected = 0.5 * whitener.running_mean + 0.5 * batch.mean(dim=0)
    torch.testing.assert_close(resumed.running_mean, expected)


def test_diagonal_whitener_is_identity_before_the_first_update():
    x = _data()
    torch.testing.assert_close(DiagonalWhitener(dim=20, eps=0.0)(x), x)


@pytest.mark.parametrize("device", DEVICES)
def test_dynamic_svd_state_loads_into_fresh_module(device):
    svd = _fitted_svd().to(device)
    assert svd.components.device.type == device

    fresh = DynamicSVD(n_components=4, buffer_size=32, whitener=DiagonalWhitener(dim=20)).to(device)
    fresh.load_state_dict(svd.state_dict())
    fresh.eval()

    assert fresh.initialized
    x = _data(seed=7).to(device)
    torch.testing.assert_close(fresh(x), svd(x))


def test_dynamic_svd_loads_legacy_extra_state():
    svd = _fitted_svd()
    legacy = {k: v for k, v in svd.state_dict().items() if not k.split(".")[-1] in DynamicSVD._lazy_buffers}
    legacy["_extra_state"] = {
        "components": svd.components, "eigenvalues": svd.eigenvalues, "_R": svd._R,
    }

    fresh = DynamicSVD(n_components=4, buffer_size=32, whitener=DiagonalWhitener(dim=20))
    fresh.load_state_dict(legacy)
    fresh.eval()

    x = _data(seed=7)
    torch.testing.assert_close(fresh(x), svd(x))
    assert fresh._noise_var is None
