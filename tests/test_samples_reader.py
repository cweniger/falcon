"""
Tests for the samples_reader module.
"""
import numpy as np
import pytest
from pathlib import Path

from falcon.core.samples_reader import read_samples, SamplesReader, SampleSetReader


class TestSampleSetReader:
    def test_loads_samples_from_npz(self, tmp_path):
        """SampleSetReader loads samples from NPZ files."""
        batch_dir = tmp_path / "posterior" / "250113-1200"
        batch_dir.mkdir(parents=True)

        # Create sample files
        np.savez(batch_dir / "000000.npz", z=np.array([1.0, 2.0]), _batch="250113-1200")
        np.savez(batch_dir / "000001.npz", z=np.array([3.0, 4.0]), _batch="250113-1200")

        reader = SampleSetReader(tmp_path / "posterior")

        assert len(reader) == 2

    def test_index_access_returns_dict(self, tmp_path):
        """SampleSetReader[int] returns a dict for that sample."""
        batch_dir = tmp_path / "posterior" / "250113-1200"
        batch_dir.mkdir(parents=True)

        np.savez(batch_dir / "000000.npz", z=np.array([1.0, 2.0]), x=np.array([0.5]))

        reader = SampleSetReader(tmp_path / "posterior")
        sample = reader[0]

        assert isinstance(sample, dict)
        np.testing.assert_array_equal(sample["z"], [1.0, 2.0])
        np.testing.assert_array_equal(sample["x"], [0.5])

    def test_negative_index_access(self, tmp_path):
        """SampleSetReader supports negative indexing."""
        batch_dir = tmp_path / "posterior" / "250113-1200"
        batch_dir.mkdir(parents=True)

        np.savez(batch_dir / "000000.npz", z=np.array([1.0]))
        np.savez(batch_dir / "000001.npz", z=np.array([2.0]))

        reader = SampleSetReader(tmp_path / "posterior")

        np.testing.assert_array_equal(reader[-1]["z"], [2.0])

    def test_key_access_returns_list(self, tmp_path):
        """SampleSetReader[str] returns list of arrays for that key."""
        batch_dir = tmp_path / "posterior" / "250113-1200"
        batch_dir.mkdir(parents=True)

        np.savez(batch_dir / "000000.npz", z=np.array([1.0, 2.0]))
        np.savez(batch_dir / "000001.npz", z=np.array([3.0, 4.0]))

        reader = SampleSetReader(tmp_path / "posterior")
        z_values = reader["z"]

        assert isinstance(z_values, list)
        assert len(z_values) == 2
        np.testing.assert_array_equal(z_values[0], [1.0, 2.0])
        np.testing.assert_array_equal(z_values[1], [3.0, 4.0])

    def test_slice_access_returns_list_of_dicts(self, tmp_path):
        """SampleSetReader[slice] returns list of sample dicts."""
        batch_dir = tmp_path / "posterior" / "250113-1200"
        batch_dir.mkdir(parents=True)

        np.savez(batch_dir / "000000.npz", z=np.array([1.0]))
        np.savez(batch_dir / "000001.npz", z=np.array([2.0]))
        np.savez(batch_dir / "000002.npz", z=np.array([3.0]))

        reader = SampleSetReader(tmp_path / "posterior")
        samples = reader[0:2]

        assert isinstance(samples, list)
        assert len(samples) == 2
        np.testing.assert_array_equal(samples[0]["z"], [1.0])
        np.testing.assert_array_equal(samples[1]["z"], [2.0])

    def test_stacked_access(self, tmp_path):
        """SampleSetReader.stacked[key] returns stacked array."""
        batch_dir = tmp_path / "posterior" / "250113-1200"
        batch_dir.mkdir(parents=True)

        np.savez(batch_dir / "000000.npz", z=np.array([1.0, 2.0]))
        np.savez(batch_dir / "000001.npz", z=np.array([3.0, 4.0]))

        reader = SampleSetReader(tmp_path / "posterior")
        stacked = reader.stacked["z"]

        assert stacked.shape == (2, 2)
        np.testing.assert_array_equal(stacked[0], [1.0, 2.0])
        np.testing.assert_array_equal(stacked[1], [3.0, 4.0])

    def test_keys_property(self, tmp_path):
        """SampleSetReader.keys returns all keys across samples."""
        batch_dir = tmp_path / "posterior" / "250113-1200"
        batch_dir.mkdir(parents=True)

        np.savez(batch_dir / "000000.npz", z=np.array([1.0]), x=np.array([2.0]), _batch="250113-1200")

        reader = SampleSetReader(tmp_path / "posterior")

        # Should exclude metadata keys starting with _
        assert reader.keys == {"z", "x"}

    def test_batches_property(self, tmp_path):
        """SampleSetReader.batches returns list of batch IDs."""
        # Create two batches
        batch1 = tmp_path / "posterior" / "250113-1200"
        batch2 = tmp_path / "posterior" / "250113-1400"
        batch1.mkdir(parents=True)
        batch2.mkdir(parents=True)

        np.savez(batch1 / "000000.npz", z=np.array([1.0]))
        np.savez(batch2 / "000000.npz", z=np.array([2.0]))

        reader = SampleSetReader(tmp_path / "posterior")

        assert set(reader.batches) == {"250113-1200", "250113-1400"}

    def test_iteration(self, tmp_path):
        """SampleSetReader supports iteration over samples."""
        batch_dir = tmp_path / "posterior" / "250113-1200"
        batch_dir.mkdir(parents=True)

        np.savez(batch_dir / "000000.npz", z=np.array([1.0]))
        np.savez(batch_dir / "000001.npz", z=np.array([2.0]))

        reader = SampleSetReader(tmp_path / "posterior")
        samples = list(reader)

        assert len(samples) == 2
        np.testing.assert_array_equal(samples[0]["z"], [1.0])
        np.testing.assert_array_equal(samples[1]["z"], [2.0])

    def test_where_filters_by_batch(self, tmp_path):
        """SampleSetReader.where(batch=...) filters samples."""
        batch1 = tmp_path / "posterior" / "250113-1200"
        batch2 = tmp_path / "posterior" / "250113-1400"
        batch1.mkdir(parents=True)
        batch2.mkdir(parents=True)

        np.savez(batch1 / "000000.npz", z=np.array([1.0]))
        np.savez(batch1 / "000001.npz", z=np.array([2.0]))
        np.savez(batch2 / "000000.npz", z=np.array([3.0]))

        reader = SampleSetReader(tmp_path / "posterior")
        filtered = reader.where(batch="250113-1200")

        assert len(filtered) == 2
        np.testing.assert_array_equal(filtered[0]["z"], [1.0])
        np.testing.assert_array_equal(filtered[1]["z"], [2.0])

    def test_where_returns_same_reader_type(self, tmp_path):
        """SampleSetReader.where() returns a SampleSetReader."""
        batch_dir = tmp_path / "posterior" / "250113-1200"
        batch_dir.mkdir(parents=True)
        np.savez(batch_dir / "000000.npz", z=np.array([1.0]))

        reader = SampleSetReader(tmp_path / "posterior")
        filtered = reader.where(batch="250113-1200")

        assert isinstance(filtered, SampleSetReader)
        # All accessors should work
        assert len(filtered) == 1
        assert "z" in filtered.keys
        np.testing.assert_array_equal(filtered.stacked["z"], [[1.0]])

    def test_empty_directory(self, tmp_path):
        """SampleSetReader handles empty/missing directories gracefully."""
        reader = SampleSetReader(tmp_path / "nonexistent")

        assert len(reader) == 0
        assert reader.keys == set()
        assert reader.batches == []

    def test_heterogeneous_samples(self, tmp_path):
        """SampleSetReader handles samples with different keys."""
        batch_dir = tmp_path / "posterior" / "250113-1200"
        batch_dir.mkdir(parents=True)

        np.savez(batch_dir / "000000.npz", z=np.array([1.0]), x=np.array([2.0]))
        np.savez(batch_dir / "000001.npz", z=np.array([3.0]))  # No x

        reader = SampleSetReader(tmp_path / "posterior")

        # Key access should only return samples that have the key
        z_values = reader["z"]
        x_values = reader["x"]

        assert len(z_values) == 2
        assert len(x_values) == 1  # Only one sample has x


class TestSamplesReader:
    def test_types_property(self, tmp_path):
        """SamplesReader.types lists available sample types."""
        (tmp_path / "posterior").mkdir()
        (tmp_path / "prior").mkdir()

        reader = SamplesReader(tmp_path)

        assert set(reader.types) == {"posterior", "prior"}

    def test_attribute_access(self, tmp_path):
        """SamplesReader provides attribute access to sample types."""
        batch_dir = tmp_path / "posterior" / "250113-1200"
        batch_dir.mkdir(parents=True)
        np.savez(batch_dir / "000000.npz", z=np.array([1.0]))

        reader = SamplesReader(tmp_path)

        assert isinstance(reader.posterior, SampleSetReader)
        assert len(reader.posterior) == 1

    def test_dict_access(self, tmp_path):
        """SamplesReader provides dict-like access to sample types."""
        batch_dir = tmp_path / "posterior" / "250113-1200"
        batch_dir.mkdir(parents=True)
        np.savez(batch_dir / "000000.npz", z=np.array([1.0]))

        reader = SamplesReader(tmp_path)

        assert isinstance(reader["posterior"], SampleSetReader)


class TestReadSamples:
    def test_returns_samples_reader(self, tmp_path):
        """read_samples() returns a SamplesReader."""
        (tmp_path / "posterior").mkdir()

        result = read_samples(str(tmp_path))

        assert isinstance(result, SamplesReader)
        assert "posterior" in result.types
