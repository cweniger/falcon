#!/usr/bin/env python3
"""Plot posterior samples vs analytic posterior for the linear regression example."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

try:
    import corner
except ImportError:
    print("Install corner: pip install corner")
    raise SystemExit(1)

from falcon import load_run


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Plot posterior comparison")
    parser.add_argument(
        "--run-dir",
        default="outputs/latest",
        help="Run directory (default: outputs/latest)",
    )
    parser.add_argument(
        "--data",
        default="data/mock_data.npz",
        help="Path to mock data with analytic posterior",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output file (default: {run-dir}/posterior_corner.png)",
    )
    args = parser.parse_args()

    # Load falcon posterior samples
    run = load_run(args.run_dir)
    samples = run.samples.posterior.stacked["theta"]
    print(f"Loaded {len(samples)} posterior samples, shape: {samples.shape}")

    n_params = samples.shape[1]
    labels = [f"$\\theta_{{{i}}}$" for i in range(n_params)]

    # Load analytic posterior
    data = np.load(args.data)
    theta_true = data["theta_true"]
    mu_post = data["mu_post"]
    Sigma_post = data["Sigma_post"]

    # Draw samples from the analytic posterior for corner plot
    analytic_samples = np.random.multivariate_normal(mu_post, Sigma_post, size=len(samples))

    # Plot: analytic posterior in blue, falcon samples in red
    fig = corner.corner(
        analytic_samples,
        labels=labels,
        color="C0",
        hist_kwargs={"density": True},
        plot_datapoints=False,
        levels=(0.68, 0.95),
    )
    corner.corner(
        samples,
        fig=fig,
        color="C1",
        hist_kwargs={"density": True},
        plot_datapoints=False,
        levels=(0.68, 0.95),
    )

    # Add ground truth lines
    corner.overplot_lines(fig, theta_true, color="k", linestyle="--", linewidth=1)

    # Legend
    fig.legend(
        handles=[
            plt.Line2D([], [], color="C0", label="Analytic"),
            plt.Line2D([], [], color="C1", label="Falcon"),
            plt.Line2D([], [], color="k", linestyle="--", label="Truth"),
        ],
        loc="upper right",
        fontsize=12,
    )

    output = args.output or str(Path(args.run_dir) / "posterior_corner.png")
    fig.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
