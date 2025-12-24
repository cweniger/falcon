"""
Run reader for loading locally logged metrics.

Provides lazy-loaded access to metrics stored in chunked NPZ files.

Usage:
    run = read_run(path)
    run['z']['loss'].values     # np.array, concatenated from all chunks
    run['z']['loss'].steps      # np.array of step numbers
    run['z']['loss'].walltime   # np.array of timestamps
"""

from pathlib import Path
from typing import Optional, List
import numpy as np


class MetricReader:
    """Lazy-loaded metric from chunked NPZ files.

    Loads and concatenates all chunks for a metric on first access.
    """

    def __init__(self, metric_dir: Path):
        self.metric_dir = metric_dir
        self._values: Optional[np.ndarray] = None
        self._steps: Optional[np.ndarray] = None
        self._walltime: Optional[np.ndarray] = None
        self._loaded = False

    def _load(self):
        """Load and concatenate all chunks."""
        if self._loaded:
            return

        if not self.metric_dir.exists():
            raise FileNotFoundError(f"Metric directory not found: {self.metric_dir}")

        # Find all chunk files, sorted by name
        chunk_files = sorted(self.metric_dir.glob("chunk_*.npz"))

        if not chunk_files:
            raise FileNotFoundError(f"No chunk files found in: {self.metric_dir}")

        # Load and concatenate
        all_steps = []
        all_values = []
        all_walltime = []

        for chunk_path in chunk_files:
            data = np.load(chunk_path)
            all_steps.append(data['step'])
            all_values.append(data['value'])
            all_walltime.append(data['walltime'])

        self._steps = np.concatenate(all_steps)
        self._walltime = np.concatenate(all_walltime)

        # Handle values - could be 1D (scalars) or ND (arrays)
        if all_values[0].ndim == 1:
            self._values = np.concatenate(all_values)
        else:
            # Multi-dimensional values: concatenate along first axis
            self._values = np.concatenate(all_values, axis=0)

        self._loaded = True

    @property
    def values(self) -> np.ndarray:
        """Get all values, concatenated from chunks."""
        self._load()
        return self._values

    @property
    def steps(self) -> np.ndarray:
        """Get all step numbers, concatenated from chunks."""
        self._load()
        return self._steps

    @property
    def walltime(self) -> np.ndarray:
        """Get all timestamps, concatenated from chunks."""
        self._load()
        return self._walltime

    def __repr__(self):
        if self._loaded:
            return f"<MetricReader({self.metric_dir.name}): {len(self._values)} entries>"
        return f"<MetricReader({self.metric_dir.name}): not loaded>"


class NodeReader:
    """Dict-like access to metrics for a node.

    Provides lazy access to metrics via indexing: node['loss'], node['lr'], etc.
    """

    def __init__(self, node_dir: Path):
        self.node_dir = node_dir
        self.metrics_dir = node_dir / "metrics"
        self._metrics_cache = {}

    def __getitem__(self, metric_name: str) -> MetricReader:
        """Get a metric reader by name.

        Handles sanitized names (: replaced with _).
        """
        if metric_name not in self._metrics_cache:
            # Try exact name first, then sanitized
            metric_dir = self.metrics_dir / metric_name
            if not metric_dir.exists():
                sanitized = metric_name.replace(":", "_")
                metric_dir = self.metrics_dir / sanitized

            self._metrics_cache[metric_name] = MetricReader(metric_dir)

        return self._metrics_cache[metric_name]

    def list_metrics(self) -> List[str]:
        """List all available metrics for this node."""
        if not self.metrics_dir.exists():
            return []
        return [d.name for d in self.metrics_dir.iterdir() if d.is_dir()]

    @property
    def metrics(self) -> List[str]:
        """Alias for list_metrics()."""
        return self.list_metrics()

    def __repr__(self):
        return f"<NodeReader({self.node_dir.name}): {len(self.list_metrics())} metrics>"


class RunReader:
    """Dict-like access to nodes in a run.

    Provides lazy access to nodes via indexing: run['z'], run['theta'], etc.
    """

    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self._nodes_cache = {}

    def __getitem__(self, node_name: str) -> NodeReader:
        """Get a node reader by name."""
        if node_name not in self._nodes_cache:
            node_dir = self.run_dir / node_name
            self._nodes_cache[node_name] = NodeReader(node_dir)

        return self._nodes_cache[node_name]

    def list_nodes(self) -> List[str]:
        """List all available nodes in this run."""
        if not self.run_dir.exists():
            return []
        # A node directory should have a 'metrics' subdirectory
        nodes = []
        for d in self.run_dir.iterdir():
            if d.is_dir() and (d / "metrics").exists():
                nodes.append(d.name)
        return nodes

    @property
    def nodes(self) -> List[str]:
        """Alias for list_nodes()."""
        return self.list_nodes()

    def __repr__(self):
        return f"<RunReader({self.run_dir}): {len(self.list_nodes())} nodes>"


def read_run(path: str) -> RunReader:
    """Read a run from local metric storage.

    Args:
        path: Path to the run directory (e.g., 'outputs/run_01/graph')

    Returns:
        RunReader providing dict-like access to nodes and metrics.

    Example:
        run = read_run('outputs/run_01/graph')
        run['z']['loss'].values     # np.array of loss values
        run['z']['loss'].steps      # np.array of step numbers
        run['z'].metrics            # list of available metrics
        run.nodes                   # list of available nodes
    """
    return RunReader(Path(path))
