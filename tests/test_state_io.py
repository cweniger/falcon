"""State trees on disk (no Ray)."""

import numpy as np
import pytest

from falcon.core.state_io import (
    CHECKPOINT_NAME, has_checkpoint, load_state, read_checkpoint, save_state, write_checkpoint,
)


def _tree():
    return {
        "meta": {"round": 3, "has_best": True, "keys": ["x", "y"], "note": None, "score": np.float64(1.5)},
        "init": {"theta": np.arange(6.0).reshape(3, 2), "conditions": {"x": np.ones((3, 4), dtype=np.float32)}},
        "groups": {
            "conditional": {"flow": {"net.0.weight": np.eye(2), "count": np.array(7, dtype=np.int64)},
                            "embedding": {}},
        },
    }


def test_round_trip(tmp_path):
    path = tmp_path / "state.npz"
    save_state(path, _tree())
    loaded = load_state(path)

    assert loaded["meta"] == {"round": 3, "has_best": True, "keys": ["x", "y"], "note": None, "score": 1.5}
    np.testing.assert_array_equal(loaded["init"]["theta"], np.arange(6.0).reshape(3, 2))
    assert loaded["init"]["conditions"]["x"].dtype == np.float32
    flow = loaded["groups"]["conditional"]["flow"]
    np.testing.assert_array_equal(flow["net.0.weight"], np.eye(2))
    assert flow["count"].shape == () and flow["count"] == 7
    assert loaded["groups"]["conditional"]["embedding"] == {}
    assert not (tmp_path / "state.npz.tmp").exists()


def test_rejects_objects_and_bad_keys(tmp_path):
    with pytest.raises(TypeError, match="leaf 'a/b'"):
        save_state(tmp_path / "s.npz", {"a": {"b": object()}})
    with pytest.raises(ValueError):
        save_state(tmp_path / "s.npz", {"a/b": 1})


def test_checkpoint_prefers_best_state_and_falls_back_to_legacy(tmp_path):
    calls = []

    def legacy(node_dir):
        calls.append(node_dir)
        return {"meta": {"round": 1}}

    assert read_checkpoint(tmp_path) is None
    assert not has_checkpoint(tmp_path)
    assert read_checkpoint(tmp_path, legacy=legacy) == {"meta": {"round": 1}}

    (tmp_path / "model.pth").write_bytes(b"")
    assert has_checkpoint(tmp_path)

    write_checkpoint(tmp_path / "z", {"meta": {"round": 2}})
    assert (tmp_path / "z" / CHECKPOINT_NAME).exists()
    assert read_checkpoint(tmp_path / "z", legacy=legacy) == {"meta": {"round": 2}}
    assert len(calls) == 1
