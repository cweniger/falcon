"""Animated GIF for 13_five_components."""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from pathlib import Path


def make_gif(run_dir, out_path=None, n_frames=30, fps=4):
    run_dir = Path(run_dir)
    snap_dir = run_dir / "buffer" / "snapshots"
    files = sorted(snap_dir.glob("*.npz"))
    if not files:
        print("No snapshots found."); return

    every = max(1, len(files) // n_frames)
    groups = [files[i:i+every] for i in range(0, len(files), every)][:n_frames]

    data_dir = run_dir.parent.parent / "data"
    obs_img = z_hi_t = z_bye_t = z_dot_t = z_cross_t = z_bg_t = None
    cand = list(data_dir.glob("obs.npz"))
    if cand:
        d = np.load(cand[0])
        obs_img  = d["x"]
        z_hi_t   = d.get("z_hi_true")
        z_bye_t  = d.get("z_bye_true")
        z_dot_t  = d.get("z_dot_true")
        z_cross_t = d.get("z_cross_true")
        z_bg_t   = d.get("z_bg_true")

    frames = []
    for gi, group in enumerate(groups):
        samples = [np.load(f) for f in group]
        def g(k): return np.array([s[k] for s in samples if k in s.files])
        z_hi    = g("z_hi.value")
        z_bye   = g("z_bye.value")
        z_dot   = g("z_dot.value")
        z_cross = g("z_cross.value")
        z_bg    = g("z_bg.value")

        fig, axes = plt.subplots(2, 4, figsize=(14, 6))
        axes = axes.flatten()

        # Panel 0: observed image
        if obs_img is not None:
            axes[0].imshow(obs_img, cmap="gray"); axes[0].set_title("Observed 64×64 x")
        axes[0].axis("off")

        def sc(ax, zv, i, j, zt, color, title):
            if len(zv):
                ax.scatter(zv[:, i], zv[:, j], s=5, alpha=0.5, c=color)
                if zt is not None:
                    ax.scatter([zt[i]], [zt[j]], s=80, marker="*", c="red", zorder=5)
            ax.set_xlim(-1.3, 1.3); ax.set_ylim(-1.3, 1.3)
            ax.set_title(title, fontsize=8); ax.set_aspect("equal")

        sc(axes[1], z_hi,    0, 1, z_hi_t,    "steelblue",   "H pos (Flow)")
        sc(axes[2], z_hi,    2, 3, z_hi_t,    "deepskyblue", "I pos (Flow)")
        sc(axes[3], z_bye,   0, 1, z_bye_t,   "darkorange",  "B pos (Flow)")
        sc(axes[4], z_bye,   2, 3, z_bye_t,   "orange",      "Y pos (Flow)")
        sc(axes[5], z_bye,   4, 5, z_bye_t,   "coral",       "E pos (Flow)")
        sc(axes[6], z_dot,   0, 1, z_dot_t,   "green",       "Circle cx,cy (Flow)")
        sc(axes[7], z_cross, 0, 1, z_cross_t, "purple",      "Cross cx,cy (Gauss)")

        pct = int(100 * gi / max(1, len(groups) - 1))
        fig.suptitle(f"5-component scene — {pct}%  ({gi+1}/{len(groups)})", fontsize=11)
        fig.tight_layout()
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)[..., :3]
        frames.append(buf); plt.close(fig)

    if out_path is None:
        out_path = run_dir / "training.gif"
    imageio.mimsave(str(out_path), frames, fps=fps, loop=0)
    print(f"Saved {len(frames)}-frame GIF → {out_path}")


if __name__ == "__main__":
    run_dir = sys.argv[1] if len(sys.argv) > 1 else "output/run"
    make_gif(run_dir)
