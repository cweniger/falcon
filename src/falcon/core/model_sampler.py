"""Sampling from the best state published by the trainer, independent of the framework."""

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from falcon.core.base_estimator import BaseEstimator, Conditions, StateTree
from falcon.core.logger import info
from falcon.core.state_io import read_checkpoint

MODES = ("prior", "proposal", "posterior")


class ModelSampler:
    """Serves samples from the best state of a model.

    Until a best state has arrived, every mode samples from the prior; proposals
    keep coming from the prior for the first ``prior_rounds`` rounds.

    Args:
        model: The model (``BaseEstimator``), set up but not yet built.
    """

    def __init__(self, model: BaseEstimator):
        self.model = model
        self.prior_rounds = model.round_config.prior_rounds
        self.meta: Dict[str, Any] = {"round": 0}
        self.has_weights = False
        self.samples_served = 0

    @property
    def round(self) -> int:
        """Round the trainer is in (or was in when the state was saved)."""
        return int(self.meta.get("round", 0))

    @property
    def best_round(self) -> int:
        """Round in which the installed networks were last promoted."""
        return int(self.meta.get("best_round", self.round))

    @property
    def status(self) -> str:
        if not self.has_weights:
            return "waiting"
        return "prior" if self.round <= self.prior_rounds else "ready"

    def apply(self, tree: StateTree) -> None:
        """Install a published state (round counters, and weights if present)."""
        self.meta.update(tree.get("meta", {}))
        if "groups" not in tree:
            return
        if not self.model.built:
            if tree.get("init") is None:
                raise RuntimeError("Received network weights before the init data needed to build the networks")
            self.model.build(tree["init"])
        for name, state in tree["groups"].items():
            self.model.import_state(name, state)
        if not self.has_weights:
            info(f"Serving the best network from round {self.best_round}")
        self.has_weights = True

    def load(self, node_dir) -> bool:
        """Install the checkpoint in ``node_dir``; False if there is none."""
        tree = read_checkpoint(Path(node_dir), legacy=self.model.load_legacy)
        if tree is None:
            return False
        self.apply(tree)
        return True

    def uses_prior(self, mode: str) -> bool:
        if mode not in MODES:
            raise ValueError(f"Unknown sampling mode {mode!r}; expected one of {MODES}")
        if mode == "prior" or not self.has_weights:
            return True
        return mode == "proposal" and self.round <= self.prior_rounds

    def sample(self, mode: str, num_samples: int, conditions: Optional[Conditions] = None,
               rng: Optional[np.random.Generator] = None) -> dict:
        """Samples as ``{"value": ndarray, "log_prob": ndarray}``."""
        rng = rng if rng is not None else np.random.default_rng()
        if self.uses_prior(mode):
            if mode == "prior" and conditions:
                raise ValueError("Conditions are not supported for prior sampling.")
            result = self.model.sample_prior(rng, num_samples)
        else:
            if not conditions:
                raise ValueError(f"{mode} sampling needs the conditions {self.model.condition_keys}")
            result = self.model.sample(rng, num_samples, conditions, mode)
        self.samples_served += num_samples
        return result
