#!/usr/bin/env python
"""Print mean and std of posterior theta samples.

Usage:
    python summarize_posterior.py [--run-dir outputs/latest]
"""

import argparse
from pathlib import Path

import numpy as np
import falcon

PARAM_NAMES = ["f0", "chirp_mass", "harmonic_decay"]


def main():
    parser = argparse.ArgumentParser(description="Summarize posterior samples")
    parser.add_argument("--run-dir", type=str, default="outputs/latest")
    args = parser.parse_args()

    run = falcon.load_run(args.run_dir)
    theta = run.samples.posterior.stacked["theta"]
    print(f"Loaded {len(theta)} posterior samples from {args.run_dir}")

    # Ground truth if available
    script_dir = Path(__file__).resolve().parent
    obs_path = script_dir / "data" / "obs.npz"
    true_theta = None
    if obs_path.exists():
        true_theta = np.load(obs_path)["true_theta"]

    mean = theta.mean(axis=0)
    std = theta.std(axis=0)

    header = f"{'Parameter':<20} {'Mean':>14} {'Std':>14}"
    if true_theta is not None:
        header += f" {'True':>14} {'Bias (σ)':>10}"
    print(header)
    print("-" * len(header))

    for i, name in enumerate(PARAM_NAMES):
        line = f"{name:<20} {mean[i]:>14.6e} {std[i]:>14.6e}"
        if true_theta is not None:
            bias = (mean[i] - true_theta[i]) / std[i]
            line += f" {true_theta[i]:>14.6e} {bias:>+10.2f}"
        print(line)


if __name__ == "__main__":
    main()
