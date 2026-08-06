#!/usr/bin/env python
"""Generate animated GIF of simulated spectra from buffer dumps.

Usage: python make_animation.py [RUN_PATH]    (default: outputs/latest)

Reads the buffer dumps (samples_dir/buffer/*.npz), computes the power
spectrum of each signal via FFT, and produces an animated GIF showing
how the simulated spectra evolve as training progresses.
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# ---------------------------------------------------------------------------
# Resolve run path
# ---------------------------------------------------------------------------
run_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("outputs/latest")
buffer_dir = run_path / "samples_dir" / "buffer"
if not buffer_dir.exists():
    print(f"No buffer directory found at {buffer_dir}")
    sys.exit(1)

dump_files = sorted(buffer_dir.glob("*.npz"))
n_dumps = len(dump_files)
print(f"Found {n_dumps} buffer dumps in {buffer_dir}")

# ---------------------------------------------------------------------------
# Load a subset of dumps for the animation (evenly spaced)
# ---------------------------------------------------------------------------
N_FRAMES = min(120, n_dumps)
indices = np.linspace(0, n_dumps - 1, N_FRAMES, dtype=int)
selected_files = [dump_files[i] for i in indices]

# Load first to get signal length and set up frequency axis
d0 = np.load(selected_files[0])
signal_len = d0["x.value"].shape[0]
freqs = np.fft.rfftfreq(signal_len)  # normalized frequency

# Precompute all spectra
print("Computing power spectra...")
spectra = []
thetas = []
for f in selected_files:
    d = np.load(f)
    signal = d["x.value"]
    theta = d["theta.value"]
    # Power spectral density via FFT
    fft_vals = np.fft.rfft(signal)
    psd = np.abs(fft_vals) ** 2 / signal_len
    spectra.append(psd)
    thetas.append(theta)

spectra = np.array(spectra)
thetas = np.array(thetas)

# Find frequency range with signal (skip DC, focus on where peaks are)
# Use log-scale PSD, find range where max spectrum has interesting content
max_psd = spectra.max(axis=0)
threshold = max_psd.max() * 1e-6
interesting = np.where(max_psd > threshold)[0]
freq_lo = max(1, interesting[0] - 10)
freq_hi = min(len(freqs) - 1, interesting[-1] + 100)

# ---------------------------------------------------------------------------
# Build animation
# ---------------------------------------------------------------------------
fig, (ax_spec, ax_params) = plt.subplots(2, 1, figsize=(10, 7),
                                          gridspec_kw={"height_ratios": [3, 1]})
fig.subplots_adjust(hspace=0.35)

# Spectrum axes
ax_spec.set_xlim(freqs[freq_lo], freqs[freq_hi])
psd_slice = spectra[:, freq_lo:freq_hi]
ymin = max(psd_slice[psd_slice > 0].min() * 0.1, 1e-10)
ymax = psd_slice.max() * 10
ax_spec.set_ylim(ymin, ymax)
ax_spec.set_yscale("log")
ax_spec.set_xlabel("Normalized frequency")
ax_spec.set_ylabel("Power spectral density")
line, = ax_spec.plot([], [], lw=0.5, color="steelblue")
title = ax_spec.set_title("")

# Parameter scatter axes — show all parameters up to current frame
param_labels = [r"$f_0$", r"$\mathcal{M}$", r"$\alpha$"]
colors = ["#e74c3c", "#2ecc71", "#3498db"]
scatters = []
for i in range(3):
    ax_p = ax_params if i == 0 else ax_params.twinx()
    if i == 2:
        ax_p.spines["right"].set_position(("axes", 1.08))
    sc = ax_p.scatter([], [], s=8, alpha=0.5, color=colors[i], label=param_labels[i])
    ax_p.set_ylabel(param_labels[i], color=colors[i], fontsize=10)
    ax_p.tick_params(axis="y", labelcolor=colors[i])
    scatters.append((ax_p, sc))

ax_params.set_xlabel("Sample index")
ax_params.set_xlim(0, n_dumps)
# Set y-ranges from data
for i, (ax_p, sc) in enumerate(scatters):
    margin = (thetas[:, i].max() - thetas[:, i].min()) * 0.1 or 1e-8
    ax_p.set_ylim(thetas[:, i].min() - margin, thetas[:, i].max() + margin)

fig.legend([s for _, s in scatters], param_labels, loc="upper right",
           fontsize=8, markerscale=2)


def init():
    line.set_data([], [])
    title.set_text("")
    for _, sc in scatters:
        sc.set_offsets(np.empty((0, 2)))
    return [line, title] + [sc for _, sc in scatters]


def update(frame):
    idx = frame
    # Update spectrum
    line.set_data(freqs[freq_lo:freq_hi], spectra[idx, freq_lo:freq_hi])
    sample_idx = indices[idx]
    title.set_text(f"Sample {sample_idx}/{n_dumps - 1}  |  "
                   f"$f_0$={thetas[idx, 0]:.6e}  "
                   f"$\\mathcal{{M}}$={thetas[idx, 1]:.6f}  "
                   f"$\\alpha$={thetas[idx, 2]:.4f}")

    # Update parameter scatter — show all points up to current frame
    for i, (ax_p, sc) in enumerate(scatters):
        offsets = np.column_stack([indices[:idx + 1], thetas[:idx + 1, i]])
        sc.set_offsets(offsets)

    return [line, title] + [sc for _, sc in scatters]


print(f"Rendering {N_FRAMES} frames...")
anim = FuncAnimation(fig, update, frames=N_FRAMES, init_func=init,
                     blit=True, interval=80)

out_path = run_path / "spectra_animation.gif"
anim.save(str(out_path), writer=PillowWriter(fps=12))
print(f"Saved animation to {out_path}")
plt.close()
