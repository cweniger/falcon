"""Tests for the Batch class and batch_collate_fn."""

import pytest
import numpy as np
import torch
from unittest.mock import MagicMock, call

from falcon.core.raystore import Batch, batch_collate_fn


class TestBatch:
    """Test Batch class functionality."""

    def test_dictionary_access(self):
        """Batch should support dictionary-style access."""
        ids = np.array([0, 1, 2])
        data = {
            'theta': torch.tensor([[1.0], [2.0], [3.0]]),
            'x': torch.tensor([[10.0], [20.0], [30.0]]),
        }
        mock_dm = MagicMock()

        batch = Batch(ids, data, mock_dm)

        assert torch.equal(batch['theta'], data['theta'])
        assert torch.equal(batch['x'], data['x'])

    def test_contains(self):
        """Batch should support 'in' operator."""
        ids = np.array([0, 1])
        data = {'theta': torch.tensor([1.0, 2.0])}
        mock_dm = MagicMock()

        batch = Batch(ids, data, mock_dm)

        assert 'theta' in batch
        assert 'nonexistent' not in batch

    def test_keys(self):
        """Batch.keys() should return data keys."""
        ids = np.array([0])
        data = {'a': torch.tensor([1.0]), 'b': torch.tensor([2.0])}
        mock_dm = MagicMock()

        batch = Batch(ids, data, mock_dm)

        assert set(batch.keys()) == {'a', 'b'}

    def test_len(self):
        """len(batch) should return batch size."""
        ids = np.array([0, 1, 2, 3, 4])
        data = {'x': torch.tensor([1, 2, 3, 4, 5])}
        mock_dm = MagicMock()

        batch = Batch(ids, data, mock_dm)

        assert len(batch) == 5

    def test_get_with_default(self):
        """Batch.get() should support default values."""
        ids = np.array([0])
        data = {'theta': torch.tensor([1.0])}
        mock_dm = MagicMock()

        batch = Batch(ids, data, mock_dm)

        assert batch.get('theta') is not None
        assert batch.get('nonexistent', 'default') == 'default'
        assert batch.get('nonexistent') is None

    def test_items_and_values(self):
        """Batch should support items() and values()."""
        ids = np.array([0])
        data = {'a': torch.tensor([1.0]), 'b': torch.tensor([2.0])}
        mock_dm = MagicMock()

        batch = Batch(ids, data, mock_dm)

        assert len(list(batch.items())) == 2
        assert len(list(batch.values())) == 2

    def test_discard_with_tensor_mask(self):
        """Batch.discard() should call dataset_manager.deactivate with correct IDs."""
        ids = np.array([10, 20, 30, 40])
        data = {'x': torch.tensor([1, 2, 3, 4])}
        mock_dm = MagicMock()

        batch = Batch(ids, data, mock_dm)

        # Discard samples at indices 1 and 3 (ids 20 and 40)
        mask = torch.tensor([False, True, False, True])
        batch.discard(mask)

        mock_dm.deactivate.remote.assert_called_once_with([20, 40])

    def test_discard_with_numpy_mask(self):
        """Batch.discard() should work with numpy masks."""
        ids = np.array([100, 200, 300])
        data = {'x': torch.tensor([1, 2, 3])}
        mock_dm = MagicMock()

        batch = Batch(ids, data, mock_dm)

        mask = np.array([True, False, True])
        batch.discard(mask)

        mock_dm.deactivate.remote.assert_called_once_with([100, 300])

    def test_discard_with_no_discards(self):
        """Batch.discard() should not call deactivate if no samples to discard."""
        ids = np.array([1, 2, 3])
        data = {'x': torch.tensor([1, 2, 3])}
        mock_dm = MagicMock()

        batch = Batch(ids, data, mock_dm)

        mask = torch.tensor([False, False, False])
        batch.discard(mask)

        mock_dm.deactivate.remote.assert_not_called()

    def test_discard_with_none(self):
        """Batch.discard(None) should do nothing."""
        ids = np.array([1, 2, 3])
        data = {'x': torch.tensor([1, 2, 3])}
        mock_dm = MagicMock()

        batch = Batch(ids, data, mock_dm)
        batch.discard(None)

        mock_dm.deactivate.remote.assert_not_called()


class TestBatchCollateFn:
    """Test batch_collate_fn factory."""

    def test_collate_creates_batch(self):
        """Collate function should create Batch objects."""
        mock_dm = MagicMock()
        collate = batch_collate_fn(mock_dm)

        samples = [
            (0, {'theta': np.array([1.0, 2.0]), 'x': np.array([10.0])}),
            (1, {'theta': np.array([3.0, 4.0]), 'x': np.array([20.0])}),
        ]

        batch = collate(samples)

        assert isinstance(batch, Batch)
        assert len(batch) == 2
        assert 'theta' in batch
        assert 'x' in batch

    def test_collate_stacks_numpy_arrays(self):
        """Collate should stack numpy arrays into numpy arrays."""
        mock_dm = MagicMock()
        collate = batch_collate_fn(mock_dm)

        samples = [
            (0, {'theta': np.array([1.0, 2.0])}),
            (1, {'theta': np.array([3.0, 4.0])}),
            (2, {'theta': np.array([5.0, 6.0])}),
        ]

        batch = collate(samples)

        assert isinstance(batch['theta'], np.ndarray)
        assert batch['theta'].shape == (3, 2)
        np.testing.assert_array_equal(batch['theta'][0], np.array([1.0, 2.0]))
        np.testing.assert_array_equal(batch['theta'][2], np.array([5.0, 6.0]))

    def test_collate_stacks_tensors(self):
        """Collate should convert torch tensors to numpy arrays."""
        mock_dm = MagicMock()
        collate = batch_collate_fn(mock_dm)

        samples = [
            (0, {'x': torch.tensor([1.0])}),
            (1, {'x': torch.tensor([2.0])}),
        ]

        batch = collate(samples)

        assert isinstance(batch['x'], np.ndarray)
        assert batch['x'].shape == (2, 1)

    def test_collate_preserves_ids(self):
        """Collate should preserve sample IDs for discard."""
        mock_dm = MagicMock()
        collate = batch_collate_fn(mock_dm)

        samples = [
            (42, {'x': np.array([1.0])}),
            (99, {'x': np.array([2.0])}),
        ]

        batch = collate(samples)

        # Verify IDs are preserved by testing discard
        mask = np.array([True, False])
        batch.discard(mask)

        mock_dm.deactivate.remote.assert_called_once_with([42])
