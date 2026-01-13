"""
Unified loader for Falcon run outputs.

Provides convenient access to all outputs from a training run:
config, samples, metrics, and observations.

Usage:
    from falcon import load_run

    run = load_run('outputs/run_01')
    run.config                    # OmegaConf config
    run.samples.posterior         # SampleSetReader
    run.metrics['z']['loss']      # MetricReader
    run.observations              # Dict[str, np.ndarray]
"""

from pathlib import Path
from typing import Dict, Optional
import numpy as np

from omegaconf import OmegaConf, DictConfig

from .run_reader import RunReader
from .samples_reader import SamplesReader


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
        self._metrics: Optional[RunReader] = None
        self._observations: Optional[Dict[str, np.ndarray]] = None

    @property
    def config(self) -> DictConfig:
        """Load and return the run configuration."""
        if self._config is None:
            config_path = self.run_dir / "config.yaml"
            if not config_path.exists():
                raise FileNotFoundError(f"No config.yaml found in {self.run_dir}")
            self._config = OmegaConf.load(config_path)
        return self._config

    @property
    def samples(self) -> SamplesReader:
        """Access samples (posterior, prior, proposal)."""
        if self._samples is None:
            samples_dir = self.run_dir / "samples_dir"
            self._samples = SamplesReader(samples_dir)
        return self._samples

    @property
    def metrics(self) -> RunReader:
        """Access training metrics."""
        if self._metrics is None:
            graph_dir = self.run_dir / "graph_dir"
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

                obs_path = node_cfg.observed
                obs_path = Path(obs_path)

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
                    self._observations[node_name] = np.load(obs_path)

        return self._observations

    def __repr__(self):
        return f"<Run({self.run_dir})>"


def load_run(path: str) -> Run:
    """Load a Falcon run from a directory.

    Args:
        path: Path to the run directory (e.g., 'outputs/run_01')

    Returns:
        Run object with access to config, samples, metrics, observations.

    Example:
        run = load_run('outputs/run_01')
        run.config                    # OmegaConf config
        run.samples.posterior['z']    # List of z arrays
        run.samples.posterior.stacked['z']  # Stacked array
        run.metrics['z']['loss'].values     # Loss curve
        run.observations              # {'x': array(...)}
    """
    return Run(Path(path))
