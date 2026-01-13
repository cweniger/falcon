#!/usr/bin/env python
"""
Simple plotting script for Falcon run outputs.

Usage:
    python make_plots.py RUN_PATH
    python make_plots.py outputs/250113-1200-happy-falcon
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import falcon


def plot_corner(samples, obs=None, title="Posterior samples"):
    """Create a corner plot."""
    if samples.ndim == 1:
        samples = samples.reshape(-1, 1)

    ndim = samples.shape[1]

    fig, axes = plt.subplots(ndim, ndim, figsize=(2.5 * ndim, 2.5 * ndim))
    if ndim == 1:
        axes = np.array([[axes]])

    for i in range(ndim):
        for j in range(ndim):
            ax = axes[i, j]

            if i == j:
                ax.hist(samples[:, i], bins=30, density=True, alpha=0.7, color='steelblue')
                if obs is not None and i < len(obs):
                    ax.axvline(obs[i], color='red', linestyle='--', linewidth=2)
            elif i > j:
                ax.scatter(samples[:, j], samples[:, i], alpha=0.3, s=5, color='steelblue')
                if obs is not None and j < len(obs) and i < len(obs):
                    ax.axhline(obs[i], color='red', linestyle='--', alpha=0.5)
                    ax.axvline(obs[j], color='red', linestyle='--', alpha=0.5)
                    ax.scatter([obs[j]], [obs[i]], color='red', s=50, marker='x', zorder=10)
            else:
                ax.axis('off')

            if i == ndim - 1:
                ax.set_xlabel(f'z[{j}]')
            if j == 0 and i > 0:
                ax.set_ylabel(f'z[{i}]')

    plt.suptitle(f'{title} ({len(samples)} samples)', y=1.02)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python make_plots.py RUN_PATH")
        sys.exit(1)

    run = falcon.load_run(sys.argv[1])

    z = run.samples.posterior.stacked['z']
    obs = run.observations.get('x')

    print(f"Loaded {len(z)} posterior samples")
    if obs is not None:
        print(f"Observation: {obs}")

    plot_corner(z, obs)
