import numpy as np

import falcon.core.raystore as raystore_module
from falcon.core.raystore import DatasetManagerActor


ActorClass = DatasetManagerActor.__ray_actor_class__


def _make_actor(tmp_path, snapshot_every):
    actor = ActorClass.__new__(ActorClass)
    actor.snapshot_every = snapshot_every
    actor.snapshots_path = tmp_path / "buffer" / "snapshots"
    actor._sample_counter = 0
    return actor


def _saved_values(actor):
    files = sorted(actor.snapshots_path.glob("*.npz"))
    return [int(np.load(path)["x"][0]) for path in files]


def test_dump_store_saves_every_nth_sample(tmp_path):
    actor = _make_actor(tmp_path, snapshot_every=3)
    samples = [{"x": np.array([i])} for i in range(1, 8)]

    actor.dump_store(samples)

    assert _saved_values(actor) == [3, 6]


def test_snapshot_every_zero_disables_snapshotting(tmp_path):
    actor = _make_actor(tmp_path, snapshot_every=0)
    samples = [{"x": np.array([i])} for i in range(1, 4)]

    actor.dump_store(samples)

    assert _saved_values(actor) == []


def test_initial_and_dynamic_samples_share_snapshot_counter(tmp_path, monkeypatch):
    actor = _make_actor(tmp_path, snapshot_every=2)
    monkeypatch.setattr(raystore_module.ray, "get", lambda value: value)

    initial_samples = [{"x": np.array([1])}, {"x": np.array([2])}]
    dynamic_sample_refs = [{"x": np.array([3])}, {"x": np.array([4])}]

    actor.dump_store(initial_samples)
    actor._dump_refs(dynamic_sample_refs)

    assert _saved_values(actor) == [2, 4]
    assert actor._sample_counter == 4
