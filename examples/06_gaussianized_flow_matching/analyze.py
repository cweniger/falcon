"""Corner plot of the GaussianizedFlowMatching posterior vs the known bimodal modes.

Run from this directory after `falcon launch -c config.yml -o output/run` and
`falcon sample posterior -o output/run`. Writes output/run/posterior_corner.png
and output/run/buffer_trace.png (first latent dim of the training buffer over time).
"""
import glob
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import falcon

RUN = sys.argv[1] if len(sys.argv) > 1 else "output/run"
SHIFT, WIDTH, D = 0.5, 0.1, 5
Z = np.stack([np.load(f)["z"] for f in sorted(glob.glob(f"{RUN}/samples/posterior/*.npz"))])
x_obs = np.load("data/x_obs.npy").ravel()
true_z = np.load("data/true_z.npy").ravel()
print(f"posterior samples: {Z.shape}\nx_obs = {x_obs.round(3)}\ntrue_z= {true_z.round(3)}")

print("\nper-dim bimodal recovery (expected modes x_obs +/- 0.5, width 0.1):")
for d in range(D):
    lo, hi = x_obs[d] - SHIFT, x_obs[d] + SHIFT
    col = Z[:, d]
    nl, nh = col[np.abs(col - lo) < np.abs(col - hi)], col[np.abs(col - hi) <= np.abs(col - lo)]
    print(f"  z{d}: ({lo:+.2f},{hi:+.2f})  lo {len(nl)/len(col):.2f}@{nl.mean():+.3f}(s{nl.std():.3f})"
          f"  hi {len(nh)/len(col):.2f}@{nh.mean():+.3f}(s{nh.std():.3f})")

lim = lambda d: (x_obs[d] - SHIFT - 0.4, x_obs[d] + SHIFT + 0.4)
fig, axes = plt.subplots(D, D, figsize=(13, 13))
for i in range(D):
    for j in range(D):
        ax = axes[i, j]
        if i < j:
            ax.axis("off"); continue
        if i == j:
            ax.hist(Z[:, i], bins=60, color="C0", alpha=0.7, density=True, range=lim(i))
            for m in (x_obs[i] - SHIFT, x_obs[i] + SHIFT): ax.axvline(m, color="red", ls="--", lw=0.9)
            ax.axvline(true_z[i], color="green", lw=1.2); ax.set_xlim(*lim(i))
        else:
            ax.scatter(Z[:, j], Z[:, i], s=4, alpha=0.2, color="C0")
            for a in (x_obs[j] - SHIFT, x_obs[j] + SHIFT):
                for b in (x_obs[i] - SHIFT, x_obs[i] + SHIFT):
                    ax.plot(a, b, "x", color="red", ms=7, mew=1.5)
            ax.plot(true_z[j], true_z[i], "*", color="green", ms=12)
            ax.set_xlim(*lim(j)); ax.set_ylim(*lim(i))
        if i == D - 1: ax.set_xlabel(f"z{j}")
        if j == 0: ax.set_ylabel(f"z{i}")
fig.suptitle("GaussianizedFlowMatching: 5D bimodal posterior\n"
             "red x = true modes (x_obs ± 0.5), green ★ = true z", fontsize=12)
fig.tight_layout()
fig.savefig(f"{RUN}/posterior_corner.png", dpi=110)
print(f"\nsaved {RUN}/posterior_corner.png")

# --- buffer trace: first latent dim of z.value over snapshot index (cf. buffer.ipynb) ---
run = falcon.load_run(RUN)
if len(run.buffer) == 0:
    print(f"no buffer snapshots in {RUN}/buffer/snapshots "
          "(set buffer.snapshot_every > 0 to enable); skipping buffer trace plot")
else:
    zb = run.buffer.stacked["z.value"]
    fig2, ax = plt.subplots(figsize=(8, 4))
    ax.plot(zb[:, 0], ".", ms=3, alpha=0.5, color="C0")
    ax.set_xlabel("buffer snapshot index")
    ax.set_ylabel("z0")
    ax.set_title("GaussianizedFlowMatching: buffer z0 over training")
    fig2.tight_layout()
    fig2.savefig(f"{RUN}/buffer_trace.png", dpi=110)
    print(f"saved {RUN}/buffer_trace.png")
