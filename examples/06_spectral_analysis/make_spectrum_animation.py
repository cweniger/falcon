#!/usr/bin/env python
"""Generate animated GIF of training progression from buffer dumps.

Usage: python make_spectrum_animation.py [RUN_PATH]    (default: outputs/latest)

Top panel: noise-free power spectrum for the current sample's theta,
with the observation's true spectrum as reference.
Bottom panels: evolving 2D scatter of theta samples with true values
and CRB error ellipses.
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Ellipse
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from chirp import chirp_signal

# ---------------------------------------------------------------------------
# Resolve run path and config
# ---------------------------------------------------------------------------
script_dir = Path(__file__).resolve().parent
run_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("outputs/latest")
buffer_dir = run_path / "samples_dir" / "buffer"
if not buffer_dir.exists():
    print(f"No buffer directory found at {buffer_dir}")
    sys.exit(1)

dump_files = sorted(buffer_dir.glob("*.npz"))
n_dumps = len(dump_files)
print(f"Found {n_dumps} buffer dumps in {buffer_dir}")

# Signal parameters from config (with defaults matching model.py)
from omegaconf import OmegaConf
cfg = OmegaConf.load(run_path / "config.yml")
sim_cfg = cfg.graph.x.simulator
T_C = float(sim_cfg.get("t_c", 1e6))
A0 = float(sim_cfg.get("A0", 5.0))
N_HARMONICS = int(sim_cfg.get("n_harmonics", 4))
N_SIG = int(sim_cfg.get("N", 100_000))
NOISE_SIGMA = float(sim_cfg.get("noise_sigma", 1.0))

# ---------------------------------------------------------------------------
# Load observation and true spectrum
# ---------------------------------------------------------------------------
obs = np.load(script_dir / "data" / "obs.npz")
true_theta = obs["true_theta"]

# Compute true (noise-free) spectrum — first call triggers JIT
print("Computing reference spectrum (JIT warmup)...")
true_signal = chirp_signal(
    f0=float(true_theta[0]), chirp_mass=float(true_theta[1]), t_c=T_C, A0=A0,
    harmonic_decay=float(true_theta[2]), n_harmonics=N_HARMONICS, N=N_SIG,
)
true_signal = np.asarray(true_signal)
freqs = np.fft.rfftfreq(N_SIG)
true_psd = np.abs(np.fft.rfft(true_signal)) ** 2 / N_SIG

# ---------------------------------------------------------------------------
# Load subset of dumps
# ---------------------------------------------------------------------------
N_FRAMES = min(200, n_dumps)
indices = np.linspace(0, n_dumps - 1, N_FRAMES, dtype=int)
selected_files = [dump_files[i] for i in indices]

print("Loading theta values...")
all_thetas = np.array([np.load(f)["theta.value"] for f in selected_files])

# ---------------------------------------------------------------------------
# Precompute all noise-free spectra
# ---------------------------------------------------------------------------
print(f"Computing {N_FRAMES} noise-free spectra...")
all_psd = np.empty((N_FRAMES, len(freqs)))
for idx in range(N_FRAMES):
    theta = all_thetas[idx]
    sig = chirp_signal(
        f0=float(theta[0]), chirp_mass=float(theta[1]), t_c=T_C, A0=A0,
        harmonic_decay=float(theta[2]), n_harmonics=N_HARMONICS, N=N_SIG,
    )
    all_psd[idx] = np.abs(np.fft.rfft(np.asarray(sig))) ** 2 / N_SIG
print("Done.")

# ---------------------------------------------------------------------------
# Find interesting frequency range
# ---------------------------------------------------------------------------
max_psd = np.maximum(all_psd.max(axis=0), true_psd)
threshold = max_psd.max() * 1e-6
interesting = np.where(max_psd > threshold)[0]
freq_lo = max(1, interesting[0] - 10)
freq_hi = min(len(freqs) - 1, interesting[-1] + 50)
freq_slice = slice(freq_lo, freq_hi)

# ---------------------------------------------------------------------------
# Try to compute CRB ellipses
# ---------------------------------------------------------------------------
cov_crb = None
try:
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from chirp import _chirp_impl

    T_OBS = 0.9 * T_C

    @jax.jit
    def compute_fisher(params, sigma):
        def signal(p):
            return _chirp_impl(p[0], p[1], T_C, A0, p[2], N_HARMONICS, N_SIG, T_OBS)
        J = jax.jacfwd(signal)(params)
        return J.T @ J / sigma ** 2

    print("Computing Fisher matrix for CRB ellipses...")
    F = np.array(compute_fisher(jnp.array(true_theta, dtype=jnp.float64), NOISE_SIGMA))
    cov_crb = np.linalg.inv(F)
except Exception as e:
    print(f"Skipping CRB ellipses: {e}")

# ---------------------------------------------------------------------------
# Build figure
# ---------------------------------------------------------------------------
PAIRS = [(0, 1), (0, 2), (1, 2)]
PAIR_LABELS = [
    (r"$f_0$", r"$\mathcal{M}$"),
    (r"$f_0$", r"$\alpha$"),
    (r"$\mathcal{M}$", r"$\alpha$"),
]

fig = plt.figure(figsize=(14, 8))
gs = fig.add_gridspec(2, 3, height_ratios=[1.2, 1], hspace=0.35, wspace=0.35,
                      left=0.07, right=0.97, top=0.93, bottom=0.08)

# -- Top: spectrum (full width) --
ax_spec = fig.add_subplot(gs[0, :])
ax_spec.set_xlim(freqs[freq_lo], freqs[freq_hi])

psd_slice = np.concatenate([all_psd[:, freq_slice], true_psd[freq_slice][None, :]])
ymin = max(psd_slice[psd_slice > 0].min() * 0.3, 1e-12)
ymax = psd_slice.max() * 5
ax_spec.set_ylim(ymin, ymax)
ax_spec.set_yscale("log")
ax_spec.set_xlabel("Normalized frequency")
ax_spec.set_ylabel("Power spectral density")

# Reference spectrum (true theta)
ax_spec.plot(freqs[freq_slice], true_psd[freq_slice], color="k",
             lw=1.0, alpha=0.4, label="True signal", zorder=1)
# Current sample spectrum (animated)
line, = ax_spec.plot([], [], lw=1.0, color="C1", zorder=2, label="Sample")
ax_spec.legend(loc="upper right", fontsize=9)
title = ax_spec.set_title("")

# -- Bottom: parameter scatter plots --
ax_params = [fig.add_subplot(gs[1, i]) for i in range(3)]


def draw_crb_ellipse(ax, pair_idx):
    if cov_crb is None:
        return
    i, j = PAIRS[pair_idx]
    sub_cov = cov_crb[np.ix_([i, j], [i, j])]
    eigvals, eigvecs = np.linalg.eigh(sub_cov)
    angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
    for n_sigma in [1, 2]:
        w = 2 * n_sigma * np.sqrt(eigvals[0])
        h = 2 * n_sigma * np.sqrt(eigvals[1])
        ell = Ellipse(xy=(true_theta[i], true_theta[j]), width=w, height=h,
                      angle=angle, edgecolor="C0", facecolor="none",
                      ls="--", lw=1.2, alpha=0.6)
        ax.add_patch(ell)


param_scatters = []
for pi, ax in enumerate(ax_params):
    i, j = PAIRS[pi]
    sc = ax.scatter([], [], s=4, alpha=0.4, color="C1", rasterized=True)
    ax.axvline(true_theta[i], color="k", ls=":", lw=0.8, alpha=0.5)
    ax.axhline(true_theta[j], color="k", ls=":", lw=0.8, alpha=0.5)
    ax.plot(true_theta[i], true_theta[j], "k+", ms=10, mew=1.5, zorder=5)
    draw_crb_ellipse(ax, pi)
    ax.set_xlabel(PAIR_LABELS[pi][0])
    ax.set_ylabel(PAIR_LABELS[pi][1])

    margin_x = max((all_thetas[:, i].max() - all_thetas[:, i].min()) * 0.15, 1e-10)
    margin_y = max((all_thetas[:, j].max() - all_thetas[:, j].min()) * 0.15, 1e-10)
    ax.set_xlim(all_thetas[:, i].min() - margin_x, all_thetas[:, i].max() + margin_x)
    ax.set_ylim(all_thetas[:, j].min() - margin_y, all_thetas[:, j].max() + margin_y)
    param_scatters.append(sc)

# ---------------------------------------------------------------------------
# Animation
# ---------------------------------------------------------------------------

def init():
    line.set_data([], [])
    for sc in param_scatters:
        sc.set_offsets(np.empty((0, 2)))
    title.set_text("")
    return [line, title] + param_scatters


def update(frame):
    # Spectrum
    line.set_data(freqs[freq_slice], all_psd[frame, freq_slice])

    # Parameter scatter — accumulate
    for pi, sc in enumerate(param_scatters):
        i, j = PAIRS[pi]
        offsets = np.column_stack([all_thetas[:frame + 1, i],
                                   all_thetas[:frame + 1, j]])
        sc.set_offsets(offsets)

    theta = all_thetas[frame]
    title.set_text(
        f"Sample {indices[frame]}/{n_dumps - 1}    "
        f"$f_0$={theta[0]:.6e}   "
        f"$\\mathcal{{M}}$={theta[1]:.6f}   "
        f"$\\alpha$={theta[2]:.4f}"
    )
    return [line, title] + param_scatters


print(f"Rendering {N_FRAMES} frames...")
anim = FuncAnimation(fig, update, frames=N_FRAMES, init_func=init,
                     blit=True, interval=60)

out_path = run_path / "animation.gif"
anim.save(str(out_path), writer=PillowWriter(fps=15))
print(f"Saved animation to {out_path}")
plt.close()
