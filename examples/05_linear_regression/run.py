#!/usr/bin/env python3
"""Run the linear regression example and validate against the analytic posterior.

Steps:
1. Generate mock observation data (if needed)
2. Run falcon launch (trains model and auto-generates posterior samples)
3. Print posterior mean/std vs analytic solution
4. Save corner plot, buffer std plot, and loss plot to the run directory

Model: y = Phi @ theta + noise
  - Phi[i, k] = sin((k+1) * x_i), x in [0, 2*pi), 1000 bins, 10 parameters
  - Prior: theta ~ N(0, I)
  - Noise: N(0, 0.1^2 * I)
  - Analytic posterior: theta | y ~ N(mu_post, Sigma_post)
"""

import subprocess
import sys
from pathlib import Path
import argparse

import numpy as np
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description="Run the 05_linear_regression falcon example.")
    parser.add_argument("-o", "--output", default="output/run", help="Output directory (default: output/run)")
    parser.add_argument("-c", "--config", default=None, help="Config file (default: config.yml)")
    return parser.parse_args()


def run_command(cmd, description, cwd=None):
    print(f"\n{'='*60}\n{description}\n{'='*60}")
    print(f"Running: {' '.join(cmd)}\n")
    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        print(f"Command failed with return code {result.returncode}")
        return False
    return True


def generate_mock_data(script_dir):
    mock_data_path = script_dir / "data" / "mock_data.npz"
    if mock_data_path.exists():
        if "x" in np.load(mock_data_path):
            print(f"Mock data already exists: {mock_data_path}")
            return True
        print("Mock data is stale (missing key 'x'), regenerating...")
    return run_command(["python", "gen_mock_data.py"], "Generating mock data", cwd=script_dir / "data")


def print_posterior_stats(run_dir, script_dir):
    from falcon import load_run
    run = load_run(run_dir)
    samples = run.samples.posterior.stacked["theta"]

    data = np.load(script_dir / "data/mock_data.npz")
    theta_true = data["theta_true"]
    mu_post = data["mu_post"]
    marginal_std = data["marginal_std"]

    means = samples.mean(axis=0)
    stds = samples.std(axis=0)
    n_params = samples.shape[1]

    print(f"\n{'='*60}\nPosterior statistics (n={len(samples)} samples)\n{'='*60}")
    print(f"{'Param':<10} {'True':>8} {'Analytic':>10} {'Inferred':>10} {'Anal Std':>10} {'Inf Std':>10} {'Ratio':>8}")
    print("-" * 68)
    for k in range(n_params):
        print(f"theta[{k}]  {theta_true[k]:>8.4f} {mu_post[k]:>10.4f} {means[k]:>10.4f} "
              f"{marginal_std[k]:>10.6f} {stds[k]:>10.6f} {stds[k]/marginal_std[k]:>8.3f}")

    mean_error = np.sqrt(np.mean((means - mu_post)**2))
    std_ratio = stds / marginal_std
    print(f"\nRMSE(mean_inferred - mean_analytic): {mean_error:.6f}")
    print(f"Std ratio (inferred/analytic):        {std_ratio.mean():.4f} +/- {std_ratio.std():.4f}")

    return samples


def plot_corner(samples, script_dir, out_path):
    import corner

    data = np.load(script_dir / "data/mock_data.npz")
    theta_true = data["theta_true"]
    mu_post = data["mu_post"]
    Sigma_post = data["Sigma_post"]

    labels = [f"$\\theta_{k}$" for k in range(samples.shape[1])]
    analytic_samples = np.random.multivariate_normal(mu_post, Sigma_post, size=len(samples))

    fig = corner.corner(analytic_samples, labels=labels, color="C0",
                        plot_datapoints=False, levels=(0.68, 0.95),
                        hist_kwargs={"density": True})
    corner.corner(samples, fig=fig, color="C1",
                  plot_datapoints=False, levels=(0.68, 0.95),
                  hist_kwargs={"density": True})
    corner.overplot_lines(fig, theta_true, color="k", linestyle="--", linewidth=1)

    fig.legend(handles=[
        plt.Line2D([], [], color="C0", label="Analytic"),
        plt.Line2D([], [], color="C1", label="Falcon"),
        plt.Line2D([], [], color="k", ls="--", label="Truth"),
    ], loc="upper right", fontsize=10)

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_buffer_mean(run_dir, out_path, window=50):
    files = sorted((run_dir / "buffer/snapshots").glob("*.npz"))
    if not files:
        print("No buffer snapshots found.")
        return
    theta = np.stack([np.load(f)["theta.value"] for f in files])
    n, d = theta.shape
    cmap = plt.get_cmap("tab10")

    fig, ax = plt.subplots(figsize=(12, 4))
    for i in range(d):
        means = [theta[max(0, j - window):j, i].mean() for j in range(window, n)]
        ax.plot(range(window, n), means, color=cmap(i % 10), alpha=0.7, lw=0.8, label=f"$\\theta_{i}$")
    ax.axhline(0, color="k", lw=0.5, ls="--")
    ax.set_xlabel("Sample index")
    ax.set_ylabel(f"Rolling mean (window={window})")
    ax.set_title("Rolling mean of buffer samples per parameter")
    ax.legend(ncol=5, fontsize=7)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


def plot_buffer(run_dir, out_scatter, out_std, window=50):
    files = sorted((run_dir / "buffer/snapshots").glob("*.npz"))
    if not files:
        print("No buffer snapshots found.")
        return
    theta = np.stack([np.load(f)["theta.value"] for f in files])
    n, d = theta.shape
    cmap = plt.get_cmap("tab10")

    fig, ax = plt.subplots(figsize=(12, 4))
    for i in range(d):
        ax.scatter(range(n), theta[:, i], s=2, alpha=0.4, color=cmap(i % 10), label=f"$\\theta_{i}$")
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Parameter value")
    ax.set_title(f"Buffer samples over time (n={n})")
    ax.legend(ncol=5, fontsize=7)
    plt.tight_layout()
    plt.savefig(out_scatter, dpi=150)
    plt.close()
    print(f"Saved: {out_scatter}")

    fig, ax = plt.subplots(figsize=(12, 4))
    for i in range(d):
        stds = [theta[max(0, j - window):j, i].std() for j in range(window, n)]
        ax.plot(range(window, n), np.log10(np.array(stds) + 1e-10),
                color=cmap(i % 10), alpha=0.7, label=f"$\\theta_{i}$")
    ax.set_xlabel("Sample index")
    ax.set_ylabel(f"log10 std (rolling window={window})")
    ax.set_title("Rolling std of buffer samples")
    ax.legend(ncol=5, fontsize=7)
    plt.tight_layout()
    plt.savefig(out_std, dpi=150)
    plt.close()
    print(f"Saved: {out_std}")


def plot_loss(run_dir, out_path):
    base = run_dir / "graph/theta/metrics"

    def load_metric(key):
        chunks = sorted((base / key).glob("*.npz"))
        if not chunks:
            return None, None
        steps = np.concatenate([np.load(c)["step"] for c in chunks])
        values = np.concatenate([np.load(c)["value"] for c in chunks])
        return steps, values

    train_steps, train_loss = load_metric("train/loss")
    val_steps, val_loss = load_metric("val/loss")
    if train_steps is None:
        print("No loss metrics found.")
        return

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(train_steps, train_loss, color="tab:blue", alpha=0.6, lw=0.8, label="train")
    if val_steps is not None and len(val_steps) > 1 and len(train_steps) > 0:
        scale = train_steps[-1] / val_steps[-1]
        val_steps_scaled = val_steps * scale
    else:
        val_steps_scaled = val_steps
    ax.plot(val_steps_scaled, val_loss, color="tab:orange", lw=1.5, label="val")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Train and validation loss")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


def main():
    args = parse_args()
    script_dir = Path(__file__).parent
    run_dir = script_dir / args.output

    if not generate_mock_data(script_dir):
        sys.exit(1)

    cmd = ["falcon", "launch", f"--output={args.output}"]
    if args.config:
        cmd += [f"--config={args.config}"]
    if not run_command(cmd, "Running falcon launch (trains model + auto-samples posterior)", cwd=script_dir):
        sys.exit(1)

    samples = print_posterior_stats(run_dir, script_dir)
    if samples is None:
        sys.exit(1)

    plot_corner(samples, script_dir, run_dir / "corner.png")
    plot_buffer_mean(run_dir, run_dir / "buffer_mean.png")
    plot_buffer(run_dir, run_dir / "buffer_scatter.png", run_dir / "buffer_std.png")
    plot_loss(run_dir, run_dir / "loss.png")


if __name__ == "__main__":
    main()
