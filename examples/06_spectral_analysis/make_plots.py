#!/usr/bin/env python
"""Compare falcon posterior with Fisher/Cramér-Rao analytical bound.

Usage: python make_plots.py [RUN_PATH]    (default: outputs/latest)

Computes the 3-parameter Fisher information matrix for the EMRI signal
at the ground-truth parameters using JAX autodiff, then plots the falcon
posterior samples alongside the CRB Gaussian in a corner plot.
"""

import sys
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import corner

import falcon
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from chirp import _chirp_impl

PARAM_NAMES = [r"$f_0$", r"$\mathcal{M}$", r"$\alpha$"]
PARAM_UNITS = ["Hz", "", ""]


# =====================================================================
# Fisher information matrix (3-parameter)
# =====================================================================

def make_fisher_fn(t_c, A0, n_harmonics, N):
    """Create a JIT-compiled Fisher matrix function for given signal parameters."""
    T_OBS = 0.9 * t_c

    @jax.jit
    def compute_fisher(params, sigma):
        """Full 3x3 Fisher matrix at given parameter values.

        Parameters
        ----------
        params : jnp.ndarray, shape (3,)
            [f0, chirp_mass, harmonic_decay]
        sigma : float
            Noise standard deviation.

        Returns
        -------
        fisher : jnp.ndarray, shape (3, 3)
        """
        def signal(p):
            return _chirp_impl(p[0], p[1], t_c, A0, p[2], n_harmonics, N, T_OBS)

        J = jax.jacfwd(signal)(params)      # (N, 3)
        return J.T @ J / sigma ** 2         # (3, 3)

    return compute_fisher


# =====================================================================
# Main
# =====================================================================

def main():
    run_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("outputs/latest")
    script_dir = Path(__file__).resolve().parent

    # ── Load falcon posterior ────────────────────────────────────────
    run = falcon.load_run(run_path)
    theta = run.samples.posterior.stacked["theta"]  # (n_samples, 3)
    print(f"Loaded {len(theta)} posterior samples from {run_path}")

    # ── Ground truth ────────────────────────────────────────────────
    obs_path = script_dir / "data" / "obs.npz"
    obs = np.load(obs_path)
    true_theta = obs["true_theta"]  # (3,)
    print(f"Ground truth: f0={true_theta[0]:.6e}, M={true_theta[1]:.6f}, "
          f"alpha={true_theta[2]:.4f}")

    # ── Signal parameters from config ────────────────────────────────
    sim_cfg = run.config.graph.x.simulator
    sigma = float(sim_cfg.get("noise_sigma", 1.0))
    N = int(sim_cfg.get("N", 100_000))
    t_c = float(sim_cfg.get("t_c", 1e6))
    A0 = float(sim_cfg.get("A0", 5.0))
    n_harmonics = int(sim_cfg.get("n_harmonics", 4))

    # ── Fisher matrix ───────────────────────────────────────────────
    print(f"\nComputing Fisher matrix (JAX autodiff, N={N:,})...")
    compute_fisher = make_fisher_fn(t_c, A0, n_harmonics, N)
    params_jax = jnp.array(true_theta, dtype=jnp.float64)
    F = np.array(compute_fisher(params_jax, sigma))
    cov_crb = np.linalg.inv(F)
    std_crb = np.sqrt(np.diag(cov_crb))

    print(f"\nFisher matrix:\n{F}")
    print(f"\nCRB covariance:\n{cov_crb}")

    # ── Comparison table ────────────────────────────────────────────
    std_post = theta.std(axis=0)
    mean_post = theta.mean(axis=0)

    print(f"\n{'Parameter':<20} {'True':>12} {'Post. mean':>12} "
          f"{'Post. std':>12} {'CRB std':>12} {'Ratio':>8}")
    print("-" * 80)
    names_plain = ["f0", "chirp_mass", "harmonic_decay"]
    for i, name in enumerate(names_plain):
        ratio = std_post[i] / std_crb[i]
        print(f"{name:<20} {true_theta[i]:>12.6e} {mean_post[i]:>12.6e} "
              f"{std_post[i]:>12.6e} {std_crb[i]:>12.6e} {ratio:>8.2f}x")

    # ── Draw CRB Gaussian samples for overlay ───────────────────────
    rng = np.random.default_rng(0)
    crb_samples = rng.multivariate_normal(true_theta, cov_crb, size=len(theta))

    # ── Corner plot ─────────────────────────────────────────────────
    fig = corner.corner(
        crb_samples,
        labels=PARAM_NAMES,
        color="C0",
        hist_kwargs=dict(density=True, alpha=0.4),
        plot_datapoints=False,
        plot_density=False,
        levels=(0.68, 0.95),
        truths=true_theta,
        truth_color="k",
        show_titles=False,
    )

    corner.corner(
        theta,
        fig=fig,
        color="C1",
        hist_kwargs=dict(density=True, alpha=0.6),
        plot_datapoints=False,
        plot_density=False,
        levels=(0.68, 0.95),
    )

    # Legend
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([], [], color="C0", label="CRB (Fisher)"),
        Line2D([], [], color="C1", label="Falcon posterior"),
        Line2D([], [], color="k", ls="--", label="Ground truth"),
    ]
    fig.legend(handles=legend_handles, loc="upper right", fontsize=11,
               frameon=True)

    out_path = run_path / "fisher_comparison.png"
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
