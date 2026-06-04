#!/usr/bin/env python3
"""Run the Gaussian posterior example and analyze results.

Steps:
1. Generate mock observation data (if needed)
2. Run falcon launch (trains model and auto-generates posterior samples)
3. Print posterior mean/std vs analytical expectation
4. Save corner plot, buffer scatter, buffer std, and loss plots to the run dir
"""

import subprocess
import sys
from pathlib import Path

import argparse

import numpy as np
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description="Run the 04_gaussian falcon example.")
    parser.add_argument("-o", "--output", default="outputs/run", help="Output directory (default: outputs/run)")
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
        print(f"Mock data already exists: {mock_data_path}")
        return True
    return run_command(["python", "gen_mock_data.py"], "Generating mock data", cwd=script_dir / "data")


def print_posterior_stats(run_dir, script_dir):
    files = sorted((run_dir / "samples/posterior").glob("*.npz"))
    if not files:
        print("No posterior samples found.")
        return None

    z = np.stack([np.load(f)["z"] for f in files])
    x_obs = np.load(script_dir / "data/mock_data.npz")["x"]
    sigma = 1e-6
    z_true = np.log(x_obs)
    std_expected = sigma / x_obs

    print(f"\n{'='*60}\nPosterior statistics (n={len(z)} samples)\n{'='*60}")
    print(f"{'':12} {'z[0]':>14} {'z[1]':>14} {'z[2]':>14}")
    print(f"{'true':12} {z_true[0]:>14.6f} {z_true[1]:>14.6f} {z_true[2]:>14.6f}")
    print(f"{'mean':12} {z.mean(0)[0]:>14.6e} {z.mean(0)[1]:>14.6e} {z.mean(0)[2]:>14.6e}")
    print(f"{'std':12} {z.std(0)[0]:>14.6e} {z.std(0)[1]:>14.6e} {z.std(0)[2]:>14.6e}")
    print(f"{'std expected':12} {std_expected[0]:>14.6e} {std_expected[1]:>14.6e} {std_expected[2]:>14.6e}")
    print(f"{'ratio':12} {z.std(0)[0]/std_expected[0]:>14.4f} {z.std(0)[1]/std_expected[1]:>14.4f} {z.std(0)[2]/std_expected[2]:>14.4f}")
    return z, z_true


def plot_corner(z, z_true, out_path):
    d = z.shape[1]
    labels = [f"z[{i}]" for i in range(d)]
    fig, axes = plt.subplots(d, d, figsize=(7, 7))
    for i in range(d):
        for j in range(d):
            ax = axes[i, j]
            if j > i:
                ax.set_visible(False)
            elif i == j:
                ax.hist(z[:, i], bins=50, color="steelblue", alpha=0.7, density=True)
                ax.axvline(z_true[i], color="red", lw=1.5, ls="--")
                ax.set_yticks([])
            else:
                ax.scatter(z[:, j], z[:, i], s=1, alpha=0.3, color="steelblue", rasterized=True)
                ax.axvline(z_true[j], color="red", lw=1, ls="--")
                ax.axhline(z_true[i], color="red", lw=1, ls="--")
            if i == d - 1:
                ax.set_xlabel(labels[j])
            else:
                ax.set_xticklabels([])
            if j == 0 and i > 0:
                ax.set_ylabel(labels[i])
            else:
                ax.set_yticklabels([])
    fig.suptitle("Posterior corner plot", y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_buffer(run_dir, out_scatter, out_std, window=100):
    files = sorted((run_dir / "buffer/snapshots").glob("*.npz"))
    if not files:
        print("No buffer snapshots found.")
        return
    z = np.stack([np.load(f)["z.value"] for f in files])
    n = len(z)
    colors = ["tab:red", "tab:blue", "tab:green"]

    fig, ax = plt.subplots(figsize=(12, 4))
    for i, c in enumerate(colors):
        ax.scatter(range(n), z[:, i], s=4, alpha=0.5, color=c, label=f"z[{i}]")
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Parameter value")
    ax.set_title(f"Buffer samples over time (n={n})")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_scatter, dpi=150)
    plt.close()
    print(f"Saved: {out_scatter}")

    fig, ax = plt.subplots(figsize=(12, 4))
    for i, c in enumerate(colors):
        stds = [z[max(0, j - window):j, i].std() for j in range(window, n)]
        ax.plot(range(window, n), np.log10(np.array(stds) + 1e-10), color=c, label=f"z[{i}]")
    ax.set_xlabel("Sample index")
    ax.set_ylabel(f"log10 std (rolling window={window})")
    ax.set_title(f"Rolling std of buffer samples (n={n})")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_std, dpi=150)
    plt.close()
    print(f"Saved: {out_std}")


def plot_loss(run_dir, out_path):
    base = run_dir / "graph/z/metrics"

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

    val_steps_scaled = val_steps * (train_steps[-1] / val_steps[-1])

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(train_steps, train_loss, color="tab:blue", alpha=0.6, lw=0.8, label="train")
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

    print(f"Output: {run_dir}")
    print(f"Config: {args.config or 'config.yml (default)'}")

    if not generate_mock_data(script_dir):
        sys.exit(1)

    cmd = ["falcon", "launch", f"--output={args.output}"]
    if args.config:
        cmd += [f"--config={args.config}"]
    if not run_command(
        cmd,
        "Running falcon launch (trains model + auto-samples posterior)",
        cwd=script_dir,
    ):
        sys.exit(1)

    result = print_posterior_stats(run_dir, script_dir)
    if result is None:
        sys.exit(1)
    z, z_true = result

    plot_corner(z, z_true, run_dir / "corner.png")
    plot_buffer(run_dir, run_dir / "buffer_scatter.png", run_dir / "buffer_std.png")
    plot_loss(run_dir, run_dir / "loss.png")


if __name__ == "__main__":
    main()
