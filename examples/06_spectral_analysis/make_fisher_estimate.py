#!/usr/bin/env python
"""Compute Fisher/CRB posterior estimate from config and observation.

Usage:
    python make_fisher_estimate.py --obs data/obs.npz
    python make_fisher_estimate.py --config-name config2 --obs data/obs.npz

Reads signal parameters (N, t_c, A0, n_harmonics, noise_sigma) and prior
bounds from a config file, loads ground-truth parameters and observation
from an NPZ file, then computes the Fisher information matrix via JAX
autodiff to produce the Cramér-Rao bound (CRB) posterior estimate.

Outputs:
  - CRB summary table (printed)
  - Corner plot of CRB Gaussian samples with prior bounds (saved as PNG)
"""

import argparse
import sys
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import corner
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from chirp import _chirp_impl

PARAM_NAMES = [r"$f_0$", r"$\mathcal{M}$", r"$\alpha$"]
PARAM_NAMES_PLAIN = ["f0", "chirp_mass", "harmonic_decay"]


def make_fisher_fn(t_c, A0, n_harmonics, N):
    """Create a JIT-compiled Fisher matrix function for given signal parameters."""
    T_OBS = 0.9 * t_c

    @jax.jit
    def compute_fisher(params, sigma):
        def signal(p):
            return _chirp_impl(p[0], p[1], t_c, A0, p[2], n_harmonics, N, T_OBS)

        J = jax.jacfwd(signal)(params)      # (N, 3)
        return J.T @ J / sigma ** 2         # (3, 3)

    return compute_fisher


def main():
    parser = argparse.ArgumentParser(description="Compute Fisher/CRB posterior estimate")
    parser.add_argument("--config-name", type=str, default="config",
                        help="Config file stem (default: config -> config.yml)")
    parser.add_argument("--obs", type=str, required=True,
                        help="Path to observation NPZ file (must contain 'true_theta')")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    config_path = script_dir / f"{args.config_name}.yml"
    obs_path = Path(args.obs)

    if not config_path.exists():
        sys.exit(f"Config not found: {config_path}")
    if not obs_path.exists():
        sys.exit(f"Observation file not found: {obs_path}")

    # ── Load config ──────────────────────────────────────────────────
    cfg = OmegaConf.load(config_path)
    graph = cfg.graph

    # Signal params from y (Signal node) or x (Simulator node)
    if "y" in graph and "Signal" in graph.y.simulator.get("_target_", ""):
        sig_cfg = graph.y.simulator
    else:
        sig_cfg = graph.x.simulator

    N = int(sig_cfg.get("N", 100_000))
    t_c = float(sig_cfg.get("t_c", 1e6))
    A0 = float(sig_cfg.get("A0", 5.0))
    n_harmonics = int(sig_cfg.get("n_harmonics", 4))

    # Noise sigma from n (Noise node) or x (Simulator node)
    if "n" in graph:
        sigma = float(graph.n.simulator.get("sigma", 1.0))
    else:
        sigma = float(graph.x.simulator.get("noise_sigma", 1.0))

    # Prior bounds from theta
    prior_bounds = []
    for p in graph.theta.simulator.priors:
        prior_bounds.append((float(p[1]), float(p[2])))

    print(f"Config: {config_path.name}")
    print(f"Signal: N={N:,}, t_c={t_c:.0e}, A0={A0}, n_harmonics={n_harmonics}")
    print(f"Noise sigma: {sigma}")
    print(f"Prior bounds: {prior_bounds}")

    # ── Load observation ─────────────────────────────────────────────
    obs = np.load(obs_path)
    true_theta = obs["true_theta"]
    print(f"\nGround truth: f0={true_theta[0]:.6e}, M={true_theta[1]:.6f}, "
          f"alpha={true_theta[2]:.4f}")

    # ── Fisher matrix ────────────────────────────────────────────────
    print(f"\nComputing Fisher matrix (JAX autodiff, N={N:,})...")
    compute_fisher = make_fisher_fn(t_c, A0, n_harmonics, N)
    params_jax = jnp.array(true_theta, dtype=jnp.float64)
    F = np.array(compute_fisher(params_jax, sigma))
    cov_crb = np.linalg.inv(F)
    std_crb = np.sqrt(np.diag(cov_crb))

    print(f"\nFisher matrix:\n{F}")
    print(f"\nCRB covariance:\n{cov_crb}")

    # ── Summary table ────────────────────────────────────────────────
    print(f"\n{'Parameter':<20} {'True':>14} {'CRB std':>14} "
          f"{'Prior width':>14} {'Prior/CRB':>10}")
    print("-" * 76)
    for i, name in enumerate(PARAM_NAMES_PLAIN):
        pw = prior_bounds[i][1] - prior_bounds[i][0]
        ratio = pw / std_crb[i]
        print(f"{name:<20} {true_theta[i]:>14.6e} {std_crb[i]:>14.6e} "
              f"{pw:>14.6e} {ratio:>10.1f}x")

    # ── Corner plot ──────────────────────────────────────────────────
    rng = np.random.default_rng(0)
    crb_samples = rng.multivariate_normal(true_theta, cov_crb, size=10_000)

    fig = corner.corner(
        crb_samples,
        labels=PARAM_NAMES,
        color="C0",
        hist_kwargs=dict(density=True, alpha=0.6),
        plot_datapoints=False,
        plot_density=False,
        levels=(0.68, 0.95),
        truths=true_theta,
        truth_color="k",
        show_titles=True,
    )

    # Show prior bounds as shaded regions on diagonal
    axes = np.array(fig.axes).reshape(3, 3)
    for i in range(3):
        ax = axes[i, i]
        lo, hi = prior_bounds[i]
        ax.axvspan(lo, hi, alpha=0.15, color="C2", zorder=0)

    # CRB std text in upper-right area
    info_lines = [f"N = {N:,},  σ_noise = {sigma}"]
    for i, name in enumerate(PARAM_NAMES_PLAIN):
        info_lines.append(f"{name} = {true_theta[i]:.6e}  (σ_CRB = {std_crb[i]:.4e})")
    fig.text(0.98, 0.72, "\n".join(info_lines), fontsize=8, fontfamily="monospace",
             verticalalignment="top", horizontalalignment="right",
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    legend_handles = [
        Line2D([], [], color="C0", label="CRB (Fisher)"),
        Patch(facecolor="C2", alpha=0.3, label="Prior range"),
        Line2D([], [], color="k", ls="--", label="Ground truth"),
    ]
    fig.legend(handles=legend_handles, loc="upper right", fontsize=11, frameon=True)

    out_name = f"fisher_crb_{args.config_name}.png"
    out_path = script_dir / out_name
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
