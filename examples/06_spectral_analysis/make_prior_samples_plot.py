#!/usr/bin/env python
"""Plot prior samples: corner plot of theta (top) and signal traces (bottom).

Usage:
    python make_prior_samples_plot.py [--run-dir outputs/latest]

Loads prior samples from a completed falcon run and produces a single PNG with:
  - Upper half: corner plot of the 3 theta parameters (f0, chirp_mass, harmonic_decay)
  - Middle: 5 random signal traces in time domain (gray: noisy x, orange: clean y)
  - Lower: same 5 samples in Fourier space (power spectral density)
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import corner

import falcon

PARAM_NAMES = [r"$f_0$", r"$\mathcal{M}$", r"$\alpha$"]


def main():
    parser = argparse.ArgumentParser(description="Plot prior samples (theta corner + signal traces)")
    parser.add_argument("--run-dir", type=str, default="outputs/latest",
                        help="Path to the falcon run directory (default: outputs/latest)")
    args = parser.parse_args()

    run_path = Path(args.run_dir)
    script_dir = Path(__file__).resolve().parent

    # ── Load prior samples ───────────────────────────────────────────
    run = falcon.load_run(run_path)
    theta = run.samples.prior.stacked["theta"]  # (n_samples, 3)
    x_list = run.samples.prior["x"]             # list of (N,) arrays
    y_list = run.samples.prior["y"]             # list of (N,) arrays (clean signal)
    print(f"Loaded {len(theta)} prior theta samples from {run_path}")
    print(f"Loaded {len(x_list)} prior x samples, {len(y_list)} prior y samples")

    # ── Ground truth (if available) ──────────────────────────────────
    obs_path = script_dir / "data" / "obs.npz"
    true_theta = None
    if obs_path.exists():
        true_theta = np.load(obs_path)["true_theta"]

    # ── Read signal params for time axis ─────────────────────────────
    # Signal length from data; time axis from y (Signal) or x (Simulator) config
    N_sig = len(x_list[0])
    cfg_graph = run.config.graph
    sig_cfg = (cfg_graph.y.simulator if "y" in cfg_graph else cfg_graph.x.simulator)
    t_c = float(sig_cfg.get("t_c", 1e6))
    T_obs = 0.9 * t_c

    # ── Select random traces ─────────────────────────────────────────
    rng = np.random.default_rng(0)
    n_traces = min(1, len(x_list))
    indices = rng.choice(len(x_list), size=n_traces, replace=False)
    t = np.linspace(0, T_obs, N_sig)
    dt = T_obs / (N_sig - 1)
    freqs = np.fft.rfftfreq(N_sig, d=dt)

    # ── Build figure with subfigures ─────────────────────────────────
    fig = plt.figure(figsize=(8, 14))
    subfigs = fig.subfigures(3, 1, height_ratios=[3, 1.2, 1.2], hspace=0.05)

    # -- Upper: corner plot (let corner create its own axes) --
    corner.corner(
        theta,
        labels=PARAM_NAMES,
        fig=subfigs[0],
        color="C0",
        hist_kwargs=dict(density=True, alpha=0.6),
        plot_datapoints=False,
        plot_density=False,
        levels=(0.68, 0.95),
        truths=true_theta,
        truth_color="k",
        show_titles=True,
    )

    # -- Middle: time-domain traces (gray = noisy x, orange = clean y) --
    ax_time = subfigs[1].subplots()
    for idx in indices:
        ax_time.plot(t, x_list[idx], alpha=0.3, linewidth=0.4, color="gray")
    for idx in indices:
        ax_time.plot(t, y_list[idx], alpha=0.7, linewidth=0.6, color="C1")
    ax_time.set_xlabel("Time (s)")
    ax_time.set_ylabel("Amplitude")
    ax_time.set_title(f"{n_traces} random prior samples (gray: x, orange: clean signal)")

    # -- Lower: Fourier-domain (same traces) --
    ax_freq = subfigs[2].subplots()
    for idx in indices:
        amp_x = np.abs(np.fft.rfft(x_list[idx])) / N_sig
        ax_freq.plot(freqs, amp_x, alpha=0.3, linewidth=0.4, color="gray")
    for idx in indices:
        amp_y = np.abs(np.fft.rfft(y_list[idx])) / N_sig
        ax_freq.plot(freqs, amp_y, alpha=0.7, linewidth=0.6, color="C1")
    ax_freq.set_xlabel("Frequency (Hz)")
    ax_freq.set_ylabel(r"Amplitude $|A|$")
    ax_freq.set_yscale("log")
    ax_freq.set_title("Fourier amplitudes (same samples)")

    # ── Save ─────────────────────────────────────────────────────────
    out_path = run_path / "prior_samples.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
