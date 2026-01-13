#!/usr/bin/env python
"""
Simple plotting script for Falcon run outputs.

Usage:
    python make_plots.py RUN_PATH
    python make_plots.py outputs/250113-1200-happy-falcon

This script loads posterior samples using the Falcon samples reader
and generates diagnostic plots.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from falcon.core.samples_reader import read_samples


def load_observation(run_path: Path) -> np.ndarray:
    """Load the observed data from the config."""
    from omegaconf import OmegaConf

    config_path = run_path / "config.yaml"
    if not config_path.exists():
        return None

    cfg = OmegaConf.load(config_path)

    # Find observed node
    for node_name, node_cfg in cfg.graph.items():
        if "observed" in node_cfg:
            obs_path = node_cfg.observed
            # Handle relative paths (relative to example dir, not run dir)
            if not Path(obs_path).is_absolute():
                # Try relative to run_path first, then to cwd
                if (run_path / obs_path).exists():
                    obs_path = run_path / obs_path
                elif Path(obs_path).exists():
                    pass  # Keep as is
                else:
                    return None
            if Path(obs_path).exists():
                return np.load(obs_path)
    return None


def plot_posterior_corner(samples, obs: np.ndarray = None, output_path: Path = None):
    """Create a corner plot of posterior samples for node 'z'."""
    # Get available keys (excluding metadata)
    keys = samples.keys
    if not keys:
        print("No sample keys found, skipping corner plot")
        return

    # Use 'z' if available, otherwise use the first key
    if 'z' in keys:
        key = 'z'
    else:
        key = sorted(keys)[0]
        print(f"Using key '{key}' for corner plot")

    try:
        z = samples.stacked[key]
    except Exception as e:
        print(f"Could not stack samples for '{key}': {e}")
        # Fall back to list and try to stack manually
        z_list = samples[key]
        if not z_list:
            print(f"No samples found for key '{key}'")
            return
        z = np.array(z_list)

    if z.ndim == 1:
        z = z.reshape(-1, 1)

    ndim = z.shape[1]
    print(f"Plotting {len(z)} samples with {ndim} dimensions")

    fig, axes = plt.subplots(ndim, ndim, figsize=(2.5 * ndim, 2.5 * ndim))

    # Handle single dimension case
    if ndim == 1:
        axes = np.array([[axes]])

    for i in range(ndim):
        for j in range(ndim):
            ax = axes[i, j]

            if i == j:
                # Diagonal: histogram
                ax.hist(z[:, i], bins=30, density=True, alpha=0.7, color='steelblue')
                if obs is not None and i < len(obs):
                    ax.axvline(obs[i], color='red', linestyle='--', linewidth=2, label='true')
            elif i > j:
                # Lower triangle: scatter
                ax.scatter(z[:, j], z[:, i], alpha=0.3, s=5, color='steelblue')
                if obs is not None and j < len(obs) and i < len(obs):
                    ax.axhline(obs[i], color='red', linestyle='--', alpha=0.5)
                    ax.axvline(obs[j], color='red', linestyle='--', alpha=0.5)
                    ax.scatter([obs[j]], [obs[i]], color='red', s=50, marker='x', zorder=10)
            else:
                # Upper triangle: empty
                ax.axis('off')

            # Labels
            if i == ndim - 1:
                ax.set_xlabel(f'{key}[{j}]')
            if j == 0 and i > 0:
                ax.set_ylabel(f'{key}[{i}]')

    plt.suptitle(f'Posterior samples ({len(z)} samples)', y=1.02)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved corner plot to {output_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Generate plots from Falcon run outputs')
    parser.add_argument('run_path', type=str, help='Path to the run directory')
    parser.add_argument('--output', '-o', type=str, help='Output file path (default: show plot)')
    args = parser.parse_args()

    run_path = Path(args.run_path)
    if not run_path.exists():
        print(f"Error: Run path does not exist: {run_path}")
        sys.exit(1)

    print(f"Loading results from: {run_path}")

    # Determine samples directory
    samples_dir = run_path / "samples_dir"
    if not samples_dir.exists():
        print(f"Error: No samples_dir found at {samples_dir}")
        print("Run 'falcon sample posterior' first.")
        sys.exit(1)

    # Load samples
    samples = read_samples(str(samples_dir))
    print(f"Available sample types: {samples.types}")

    if 'posterior' not in samples.types:
        print("Error: No posterior samples found")
        print("Run 'falcon sample posterior' first.")
        sys.exit(1)

    posterior = samples.posterior
    print(f"Posterior: {posterior}")
    print(f"  Batches: {posterior.batches}")
    print(f"  Keys: {posterior.keys}")
    print(f"  Samples: {len(posterior)}")

    # Load observation (true values)
    obs = load_observation(run_path)
    if obs is not None:
        print(f"Loaded observation with shape: {obs.shape}")

    # Determine output path
    output_path = Path(args.output) if args.output else None

    # Plot
    plot_posterior_corner(posterior, obs, output_path)


if __name__ == '__main__':
    main()
