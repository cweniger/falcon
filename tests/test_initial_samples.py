"""Tests for loading initial samples from NPZ prior directories."""

import numpy as np
import pytest
from pathlib import Path

from falcon.core.samples_reader import SampleSetReader


def _remap_npz_to_buffer(sample_dict):
    """Replicate the remapping logic from DatasetManagerActor.load_initial_samples."""
    remapped = {}
    for key, value in sample_dict.items():
        if key.startswith("_"):
            continue
        remapped[f"{key}.value"] = value
        remapped[f"{key}.log_prob"] = np.float64(0.0)
    return remapped


def _load_initial_samples(path):
    """Replicate the full load logic from DatasetManagerActor.load_initial_samples."""
    reader = SampleSetReader(Path(path))
    samples = []
    for sample_dict in reader:
        samples.append(_remap_npz_to_buffer(sample_dict))
    return samples


class TestInitialSamplesLoading:
    """Test the NPZ-based initial sample loading pipeline."""

    def _create_prior_samples(self, tmp_path, n_samples=5):
        """Create a realistic prior sample directory structure (flat)."""
        prior_dir = tmp_path / "prior"
        prior_dir.mkdir(parents=True)

        for i in range(n_samples):
            np.savez(
                prior_dir / f"{i:06d}.npz",
                z=np.random.randn(3),
                x=np.random.randn(10),
            )
        return prior_dir

    def test_loads_correct_number_of_samples(self, tmp_path):
        """Loading returns the correct number of samples."""
        prior_dir = self._create_prior_samples(tmp_path, n_samples=7)
        samples = _load_initial_samples(prior_dir)
        assert len(samples) == 7

    def test_keys_are_remapped(self, tmp_path):
        """NPZ keys are remapped to dotted buffer format."""
        prior_dir = tmp_path / "prior"
        prior_dir.mkdir(parents=True)
        np.savez(prior_dir / "000000.npz", z=np.array([1.0, 2.0]), x=np.array([0.5]))

        samples = _load_initial_samples(prior_dir)
        sample = samples[0]

        assert "z.value" in sample
        assert "z.log_prob" in sample
        assert "x.value" in sample
        assert "x.log_prob" in sample

    def test_values_preserved(self, tmp_path):
        """Original array values are preserved in remapped keys."""
        prior_dir = tmp_path / "prior"
        prior_dir.mkdir(parents=True)
        np.savez(
            prior_dir / "000000.npz",
            z=np.array([1.0, 2.0, 3.0]),
            x=np.array([10.0, 20.0]),
        )

        samples = _load_initial_samples(prior_dir)
        sample = samples[0]

        np.testing.assert_array_equal(sample["z.value"], [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(sample["x.value"], [10.0, 20.0])

    def test_log_prob_is_zero(self, tmp_path):
        """Log probabilities are set to 0.0 (uniform weight for prior samples)."""
        prior_dir = tmp_path / "prior"
        prior_dir.mkdir(parents=True)
        np.savez(prior_dir / "000000.npz", z=np.array([1.0]), x=np.array([2.0]))

        samples = _load_initial_samples(prior_dir)
        sample = samples[0]

        assert sample["z.log_prob"] == np.float64(0.0)
        assert sample["x.log_prob"] == np.float64(0.0)
        assert isinstance(sample["z.log_prob"], np.float64)

    def test_metadata_keys_skipped(self, tmp_path):
        """Keys starting with _ (like _batch) are excluded from remapped output."""
        prior_dir = tmp_path / "prior"
        prior_dir.mkdir(parents=True)
        np.savez(
            prior_dir / "000000.npz",
            z=np.array([1.0]),
            _extra="metadata",
        )

        samples = _load_initial_samples(prior_dir)
        sample = samples[0]

        assert "z.value" in sample
        # Metadata keys must not appear
        assert "_extra" not in sample
        assert "_extra.value" not in sample
        # Only z.value and z.log_prob
        assert set(sample.keys()) == {"z.value", "z.log_prob"}

    def test_empty_directory_returns_empty(self, tmp_path):
        """An empty or missing directory returns no samples."""
        samples = _load_initial_samples(tmp_path / "nonexistent")
        assert samples == []

    def test_no_raw_npz_keys_leak(self, tmp_path):
        """No raw NPZ keys (without .value/.log_prob suffix) appear in output."""
        prior_dir = tmp_path / "prior"
        prior_dir.mkdir(parents=True)
        np.savez(prior_dir / "000000.npz", theta=np.array([1.0]), x=np.array([2.0]))

        samples = _load_initial_samples(prior_dir)
        sample = samples[0]

        for key in sample:
            assert "." in key, f"Key '{key}' missing dotted suffix"
