"""
Tests for the run_reader module.
"""
import numpy as np
import pytest
from pathlib import Path

from falcon.core.run_reader import read_run, RunReader, NodeReader, MetricReader


class TestMetricReader:
    def test_loads_single_chunk(self, tmp_path):
        """MetricReader loads data from a single chunk file."""
        metric_dir = tmp_path / "loss"
        metric_dir.mkdir()

        # Create a chunk file
        steps = np.array([0, 1, 2], dtype=np.int64)
        values = np.array([1.0, 0.5, 0.3], dtype=np.float64)
        walltime = np.array([100.0, 101.0, 102.0], dtype=np.float64)
        np.savez(metric_dir / "chunk_000000.npz", step=steps, value=values, walltime=walltime)

        reader = MetricReader(metric_dir)

        np.testing.assert_array_equal(reader.steps, steps)
        np.testing.assert_array_equal(reader.values, values)
        np.testing.assert_array_equal(reader.walltime, walltime)

    def test_loads_multiple_chunks(self, tmp_path):
        """MetricReader concatenates data from multiple chunks."""
        metric_dir = tmp_path / "loss"
        metric_dir.mkdir()

        # Create two chunk files
        np.savez(
            metric_dir / "chunk_000000.npz",
            step=np.array([0, 1], dtype=np.int64),
            value=np.array([1.0, 0.8], dtype=np.float64),
            walltime=np.array([100.0, 101.0], dtype=np.float64),
        )
        np.savez(
            metric_dir / "chunk_000001.npz",
            step=np.array([2, 3], dtype=np.int64),
            value=np.array([0.5, 0.3], dtype=np.float64),
            walltime=np.array([102.0, 103.0], dtype=np.float64),
        )

        reader = MetricReader(metric_dir)

        np.testing.assert_array_equal(reader.steps, [0, 1, 2, 3])
        np.testing.assert_array_equal(reader.values, [1.0, 0.8, 0.5, 0.3])
        np.testing.assert_array_equal(reader.walltime, [100.0, 101.0, 102.0, 103.0])

    def test_loads_array_values(self, tmp_path):
        """MetricReader handles multi-dimensional values."""
        metric_dir = tmp_path / "embedding"
        metric_dir.mkdir()

        values = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        np.savez(
            metric_dir / "chunk_000000.npz",
            step=np.array([0, 1], dtype=np.int64),
            value=values,
            walltime=np.array([100.0, 101.0], dtype=np.float64),
        )

        reader = MetricReader(metric_dir)

        assert reader.values.shape == (2, 2)
        np.testing.assert_array_equal(reader.values, values)

    def test_raises_on_missing_directory(self, tmp_path):
        """MetricReader raises FileNotFoundError for missing directory."""
        reader = MetricReader(tmp_path / "nonexistent")

        with pytest.raises(FileNotFoundError):
            _ = reader.values

    def test_raises_on_empty_directory(self, tmp_path):
        """MetricReader raises FileNotFoundError for empty directory."""
        metric_dir = tmp_path / "empty"
        metric_dir.mkdir()

        reader = MetricReader(metric_dir)

        with pytest.raises(FileNotFoundError):
            _ = reader.values


class TestNodeReader:
    def test_accesses_metrics(self, tmp_path):
        """NodeReader provides dict-like access to metrics."""
        node_dir = tmp_path / "z"
        metrics_dir = node_dir / "metrics" / "loss"
        metrics_dir.mkdir(parents=True)

        np.savez(
            metrics_dir / "chunk_000000.npz",
            step=np.array([0], dtype=np.int64),
            value=np.array([0.5], dtype=np.float64),
            walltime=np.array([100.0], dtype=np.float64),
        )

        reader = NodeReader(node_dir)

        assert reader["loss"].values[0] == 0.5

    def test_lists_metrics(self, tmp_path):
        """NodeReader.metrics lists available metrics."""
        node_dir = tmp_path / "z"
        (node_dir / "metrics" / "loss").mkdir(parents=True)
        (node_dir / "metrics" / "accuracy").mkdir(parents=True)

        reader = NodeReader(node_dir)

        assert set(reader.metrics) == {"loss", "accuracy"}


class TestRunReader:
    def test_accesses_nodes(self, tmp_path):
        """RunReader provides dict-like access to nodes."""
        run_dir = tmp_path
        node_dir = run_dir / "z" / "metrics" / "loss"
        node_dir.mkdir(parents=True)

        np.savez(
            node_dir / "chunk_000000.npz",
            step=np.array([0], dtype=np.int64),
            value=np.array([0.5], dtype=np.float64),
            walltime=np.array([100.0], dtype=np.float64),
        )

        reader = RunReader(run_dir)

        assert reader["z"]["loss"].values[0] == 0.5

    def test_lists_nodes(self, tmp_path):
        """RunReader.nodes lists available nodes."""
        run_dir = tmp_path
        (run_dir / "z" / "metrics").mkdir(parents=True)
        (run_dir / "theta" / "metrics").mkdir(parents=True)

        reader = RunReader(run_dir)

        assert set(reader.nodes) == {"z", "theta"}


class TestReadRun:
    def test_read_run_returns_runner_reader(self, tmp_path):
        """read_run() returns a RunReader for the given path."""
        (tmp_path / "z" / "metrics").mkdir(parents=True)

        result = read_run(str(tmp_path))

        assert isinstance(result, RunReader)
        assert "z" in result.nodes
