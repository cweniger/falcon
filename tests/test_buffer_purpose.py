"""Tests for the fixed training/validation purpose split in DatasetManagerActor."""

import numpy as np
import pytest

from falcon.core.raystore import DatasetManagerActor, SamplePurpose, SampleStatus


ActorClass = DatasetManagerActor.__ray_actor_class__


def _make_actor(min_samples=64, max_samples=128, validation_fraction=0.15,
                simulate_count=32, simulate_when_full=True):
    # Bypass __init__ (it starts an asyncio monitor task) and set the state by hand
    actor = ActorClass.__new__(ActorClass)
    actor.min_samples = min_samples
    actor.max_samples = max_samples
    actor.simulate_count = simulate_count
    actor.simulate_when_full = simulate_when_full
    actor.snapshot_every = 0
    actor.snapshots_path = None
    actor._sample_counter = 0
    actor.ray_store = []
    actor.status = np.zeros(0, dtype=int)
    actor.purpose = np.zeros(0, dtype=np.int8)
    actor.ref_counts = np.zeros(0, dtype=int)
    actor._set_validation_fraction(validation_fraction)
    return actor


def _append(actor, n):
    actor.append_refs([{"x.value": i} for i in range(n)])


def _live(actor, purpose):
    return np.isin(actor.status, [SampleStatus.ACTIVE, SampleStatus.DISFAVOURED]) & (actor.purpose == purpose)


def test_purpose_rule_for_0_15_is_3_of_every_20():
    actor = _make_actor(min_samples=64, max_samples=10_000, validation_fraction=0.15)
    _append(actor, 2000)

    val_ids = np.where(actor.purpose == SamplePurpose.VALIDATION)[0]
    assert list(val_ids[:6]) == [0, 7, 14, 20, 27, 34]
    assert len(val_ids) == 300


def test_purpose_rule_for_0_125_is_every_8th():
    actor = _make_actor(min_samples=64, max_samples=10_000, validation_fraction=0.125)
    _append(actor, 1000)

    val_ids = np.where(actor.purpose == SamplePurpose.VALIDATION)[0]
    np.testing.assert_array_equal(val_ids, np.arange(0, 1000, 8))


def test_purpose_rule_independent_of_chunking():
    one_chunk = _make_actor(max_samples=10_000)
    _append(one_chunk, 999)
    many_chunks = _make_actor(max_samples=10_000)
    for n in (1, 13, 64, 921):
        _append(many_chunks, n)

    np.testing.assert_array_equal(one_chunk.purpose, many_chunks.purpose)


def test_purpose_never_changes_through_lifecycle():
    actor = _make_actor(min_samples=64, max_samples=128)
    _append(actor, 100)
    purpose_before = actor.purpose.copy()

    actor.deactivate(list(range(0, 100, 3)))
    for _ in range(10):
        _append(actor, 32)

    np.testing.assert_array_equal(actor.purpose[:100], purpose_before)
    assert (actor.status[:100] == SampleStatus.TOMBSTONE).all()


def test_cap_tombstones_oldest_and_keeps_ratio():
    actor = _make_actor(min_samples=64, max_samples=200)
    _append(actor, 1000)

    ids_active = np.where(actor.status == SampleStatus.ACTIVE)[0]
    np.testing.assert_array_equal(ids_active, np.arange(800, 1000))
    assert (actor.status[:800] == SampleStatus.TOMBSTONE).all()
    assert int(_live(actor, SamplePurpose.VALIDATION).sum()) == 30


def test_deactivate_applies_to_both_purposes():
    actor = _make_actor()
    _append(actor, 40)

    actor.deactivate([0, 1])  # id 0 is validation, id 1 is training

    assert actor.status[0] == SampleStatus.DISFAVOURED
    assert actor.status[1] == SampleStatus.DISFAVOURED
    assert actor.purpose[0] == SamplePurpose.VALIDATION
    assert actor.purpose[1] == SamplePurpose.TRAINING


def test_floor_holds_per_purpose_under_heavy_discards():
    actor = _make_actor(min_samples=64, max_samples=128)
    _append(actor, 64)
    assert actor._min_validation == 10
    assert actor._min_training == 54

    # Discard everything, then add a few more samples
    actor.deactivate(list(range(64)))
    _append(actor, 20)

    n_val_live = int(_live(actor, SamplePurpose.VALIDATION).sum())
    n_train_live = int(_live(actor, SamplePurpose.TRAINING).sum())
    assert n_val_live == actor._min_validation
    assert n_train_live == actor._min_training
    # The disfavoured samples kept are the newest ones
    kept_disf = np.where(actor.status == SampleStatus.DISFAVOURED)[0]
    tomb = np.where(actor.status == SampleStatus.TOMBSTONE)[0]
    assert kept_disf.min() > tomb.max()


def test_checkout_refs_purposes_are_disjoint_and_cover_live():
    actor = _make_actor(min_samples=64, max_samples=128)
    _append(actor, 300)
    actor.deactivate(list(range(250, 260)))
    statuses = [SampleStatus.ACTIVE, SampleStatus.DISFAVOURED]

    train = actor.checkout_refs(statuses, ["x.value"], purpose=SamplePurpose.TRAINING)["_active_ids"]
    val = actor.checkout_refs(statuses, ["x.value"], purpose=SamplePurpose.VALIDATION)["_active_ids"]
    live = np.where(np.isin(actor.status, statuses))[0]

    assert len(np.intersect1d(train, val)) == 0
    np.testing.assert_array_equal(np.sort(np.concatenate([train, val])), live)
    assert (actor.purpose[val] == SamplePurpose.VALIDATION).all()


def test_store_stats_count_live_per_purpose():
    actor = _make_actor(min_samples=64, max_samples=128)
    _append(actor, 200)

    stats = actor.get_store_stats()
    assert stats["training"] + stats["validation"] == 128
    assert stats["validation"] == int(_live(actor, SamplePurpose.VALIDATION).sum())
    assert stats["total_length"] == 200


def test_num_initial_samples_and_resims():
    actor = _make_actor(min_samples=64, max_samples=100, simulate_count=32, simulate_when_full=False)
    assert actor.num_initial_samples() == 64

    _append(actor, 80)
    assert actor.num_resims() == 20

    actor.simulate_when_full = True
    assert actor.num_resims() == 32


@pytest.mark.parametrize("fraction", [0.0, -0.1, 0.51, 1.0, 1e-5])
def test_rejects_invalid_fraction(fraction):
    with pytest.raises(ValueError):
        _make_actor(validation_fraction=fraction)


def test_rejects_tiny_min_samples():
    with pytest.raises(ValueError):
        _make_actor(min_samples=1)
