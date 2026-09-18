"""Tests for CachedDataLoader's full-pass iteration and refresh (no Ray)."""

from unittest.mock import MagicMock

import numpy as np
import torch

from falcon.core.raystore import CachedDataLoader


def _make_loader():
    return CachedDataLoader(MagicMock(), keys=["x.value"], sample_status=[0])


def _checkout(active_ids, new_ids):
    """Fake checkout result plus tensors, where each sample's value is its id."""
    new_ids = np.asarray(new_ids, dtype=int)
    tensors = {"x.value": torch.as_tensor(new_ids, dtype=torch.float64)[:, None]} if len(new_ids) else {}
    return tensors, {"_active_ids": np.asarray(active_ids, dtype=int), "_new_ids": new_ids}


def _apply(loader, active_ids, new_ids):
    loader._apply_fetch(*_checkout(active_ids, new_ids))


def _collect(loader, **kwargs):
    ids, values, sizes = [], [], []
    for batch in loader.iter_batches(**kwargs):
        ids.extend(batch._ids.tolist())
        values.extend(batch["x.value"][:, 0].tolist())
        sizes.append(len(batch))
    return ids, values, sizes


def test_iter_batches_covers_every_sample_once():
    loader = _make_loader()
    _apply(loader, range(10), range(10))

    ids, values, sizes = _collect(loader, batch_size=4)

    assert sorted(ids) == list(range(10))
    assert values == [float(i) for i in ids]
    assert sizes == [4, 4, 2]


def test_iter_batches_after_evictions_and_free_row_reuse():
    loader = _make_loader()
    _apply(loader, range(10), range(10))
    # Evict 0-3, add 10-15: four new samples reuse freed rows, two are appended
    _apply(loader, range(4, 16), range(10, 16))

    ids, values, _ = _collect(loader, batch_size=5, shuffle=True)

    assert loader.count == 12
    assert sorted(ids) == list(range(4, 16))
    assert values == [float(i) for i in ids]


def test_iter_batches_shuffle_changes_order_but_not_content():
    torch.manual_seed(0)
    loader = _make_loader()
    _apply(loader, range(100), range(100))

    ordered, _, _ = _collect(loader, batch_size=10)
    shuffled, _, _ = _collect(loader, batch_size=10, shuffle=True)

    assert ordered != shuffled
    assert sorted(shuffled) == ordered


def test_iter_batches_drop_last():
    loader = _make_loader()
    _apply(loader, range(10), range(10))

    _, _, sizes = _collect(loader, batch_size=4, drop_last=True)
    assert sizes == [4, 4]

    # A single partial batch is still yielded, so an epoch is never empty
    _, _, sizes = _collect(loader, batch_size=32, drop_last=True)
    assert sizes == [10]


def test_iter_batches_empty_cache_yields_nothing():
    loader = _make_loader()
    assert list(loader.iter_batches(batch_size=4)) == []


def test_refresh_fetches_only_new_samples_and_applies_them():
    loader = _make_loader()
    snapshots = []

    def fake_fetch(active_ids_snapshot):
        snapshots.append(active_ids_snapshot)
        return _checkout(range(6), [i for i in range(6) if i not in active_ids_snapshot])

    loader._checkout_and_fetch = fake_fetch

    loader.refresh()
    assert loader.count == 6

    loader.refresh()
    assert loader.count == 6
    assert list(snapshots[1]) == list(range(6))  # second refresh only asks for new samples
