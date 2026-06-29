"""
Unified loader for Falcon run outputs.

Provides convenient access to all outputs from a training run:
config, samples, metrics, and observations.

Usage:
    from falcon import load_run

    run = load_run('output/run_01')
    run.config                    # OmegaConf config
    run.samples.posterior         # SampleSetReader
    run.metrics['z']['loss']      # MetricReader
    run.observations              # Dict[str, np.ndarray]
"""

from pathlib import Path
from typing import Dict, Optional
import numpy as np

from omegaconf import OmegaConf, DictConfig

from .graph import _parse_observation_path
from .run_reader import RunReader
from .samples_reader import SamplesReader, SampleSetReader


class Run:
    """Unified access to all outputs from a Falcon run.

    Attributes:
        config: The resolved OmegaConf configuration
        samples: SamplesReader for posterior/prior/proposal samples
        metrics: RunReader for training metrics
        observations: Dict of observed data arrays
    """

    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir)
        self._config: Optional[DictConfig] = None
        self._samples: Optional[SamplesReader] = None
        self._buffer: Optional[SampleSetReader] = None
        self._metrics: Optional[RunReader] = None
        self._observations: Optional[Dict[str, np.ndarray]] = None

    @property
    def config(self) -> DictConfig:
        """Load and return the run configuration."""
        if self._config is None:
            config_path = self.run_dir / "config.yml"
            if not config_path.exists():
                raise FileNotFoundError(f"No config.yml found in {self.run_dir}")
            self._config = OmegaConf.load(config_path)
        return self._config

    @property
    def samples(self) -> SamplesReader:
        """Access samples (posterior, prior, proposal)."""
        if self._samples is None:
            samples_dir = self.run_dir / "samples"
            self._samples = SamplesReader(samples_dir)
        return self._samples

    @property
    def buffer(self) -> SampleSetReader:
        """Access training buffer samples stored during training.

        Returns a SampleSetReader for buffer/snapshots.
        Samples are stored when buffer.snapshot_every > 0.

        Usage:
            run.buffer[0]           # First stored sample
            run.buffer['z']         # All z arrays
            run.buffer.stacked['z'] # Stacked array
        """
        if self._buffer is None:
            buffer_dir = self.run_dir / "buffer" / "snapshots"
            self._buffer = SampleSetReader(buffer_dir)
        return self._buffer

    @property
    def metrics(self) -> RunReader:
        """Access training metrics."""
        if self._metrics is None:
            graph_dir = self.run_dir / "graph"
            self._metrics = RunReader(graph_dir)
        return self._metrics

    @property
    def observations(self) -> Dict[str, np.ndarray]:
        """Load observed data from all observed nodes."""
        if self._observations is None:
            self._observations = {}
            cfg = self.config

            for node_name, node_cfg in cfg.graph.items():
                if "observed" not in node_cfg:
                    continue

                # Parse NPZ key extraction syntax: "file.npz['key']"
                obs_path_str = node_cfg.observed
                file_path_str, key = _parse_observation_path(obs_path_str)
                obs_path = Path(file_path_str)

                # Handle relative paths
                if not obs_path.is_absolute():
                    # Try relative to run_dir first, then cwd
                    if (self.run_dir / obs_path).exists():
                        obs_path = self.run_dir / obs_path
                    elif obs_path.exists():
                        pass
                    else:
                        continue

                if obs_path.exists():
                    data = np.load(obs_path)
                    if key is not None:
                        data = data[key]
                    elif hasattr(data, 'files') and len(data.files) == 1:
                        data = data[data.files[0]]
                    self._observations[node_name] = data

        return self._observations

    def plot_metrics(self, nodes=None, metrics=("train", "val"), figsize=None):
        """Plot training metric curves.

        Args:
            nodes: Node names to plot. Defaults to all nodes with metric data.
            metrics: Metric names to look for (e.g. ``("train", "val")``).
            figsize: Matplotlib figure size. Auto-scaled if *None*.

        Returns:
            ``matplotlib.figure.Figure``
        """
        import matplotlib.pyplot as plt

        available = self.metrics.list_nodes()
        plot_nodes = [n for n in (nodes or available) if n in available]

        if not plot_nodes:
            print("No metric data found.")
            return None

        n_cols = len(plot_nodes)
        fig, axes = plt.subplots(1, n_cols, figsize=figsize or (5 * n_cols, 4), squeeze=False)
        axes = axes[0]

        for ax, node_name in zip(axes, plot_nodes):
            node_reader = self.metrics[node_name]
            node_metrics = node_reader.list_metrics()
            plotted = False
            for metric in metrics:
                if metric in node_metrics:
                    try:
                        r = node_reader[metric]
                        ax.plot(r.steps, r.values, label=metric)
                        plotted = True
                    except Exception:
                        pass
            if not plotted:
                ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(node_name)
            ax.set_xlabel("step")
            ax.set_ylabel("loss")
            if plotted:
                ax.legend()

        fig.tight_layout()
        return fig

    def _repr_html_(self) -> str:
        """Status card for Jupyter/Colab."""
        rows = []

        # Per-node metrics summary
        metric_rows = []
        for node_name in self.metrics.list_nodes():
            node_reader = self.metrics[node_name]
            available = node_reader.list_metrics()
            final_loss = ""
            for key in ("val", "train"):
                if key in available:
                    try:
                        vals = node_reader[key].values
                        if len(vals):
                            final_loss = f"{vals[-1]:.4f}"
                    except Exception:
                        pass
                    break
            n_steps = ""
            if "epoch" in available:
                try:
                    n_steps = str(int(node_reader["epoch"].values[-1]))
                except Exception:
                    pass
            td = "padding:2px 8px"
            metric_rows.append(
                f"<tr>"
                f"<td style='{td}'><code>{node_name}</code></td>"
                f"<td style='{td};text-align:right'>{final_loss}</td>"
                f"<td style='{td};text-align:right'>{n_steps}</td>"
                f"</tr>"
            )

        if metric_rows:
            rows.append(
                "<table style='border-collapse:collapse;margin-top:6px'>"
                "<tr>"
                "<th style='text-align:left;padding:2px 8px'>node</th>"
                "<th style='text-align:right;padding:2px 8px'>final loss</th>"
                "<th style='text-align:right;padding:2px 8px'>epochs</th>"
                "</tr>"
                + "".join(metric_rows)
                + "</table>"
            )

        # Sample counts
        try:
            post = self.samples.posterior
            n_files = len(list(post._sample_dir.glob("*.npz"))) if hasattr(post, "_sample_dir") else "?"
            rows.append(f"<div style='margin-top:4px'>posterior samples: {n_files} file(s)</div>")
        except Exception:
            pass

        inner = "\n".join(rows)
        return (
            f"<div style='font-family:monospace;border:1px solid #ddd;border-radius:4px;"
            f"padding:10px 14px;background:#f8f9fa;display:inline-block;min-width:300px'>"
            f"<b>Run</b> <span style='color:#555'>{self.run_dir}</span>"
            f"{inner}"
            f"</div>"
        )

    def __repr__(self):
        return f"<Run({self.run_dir})>"


def load_run(path: str) -> Run:
    """Load a Falcon run from a directory.

    Args:
        path: Path to the run directory (e.g., 'output/run_01')

    Returns:
        Run object with access to config, samples, metrics, observations.

    Example:
        run = load_run('output/run_01')
        run.config                    # OmegaConf config
        run.samples.posterior['z']    # List of z arrays
        run.samples.posterior.stacked['z']  # Stacked array
        run.metrics['z']['loss'].values     # Loss curve
        run.observations              # {'x': array(...)}
    """
    return Run(Path(path))
