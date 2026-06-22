"""Animated GIF for 15_three_words."""
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
    obs_img = z_hi_t = z_bye_t = z_sbi_t = None
    cand = list(data_dir.glob("obs.npz"))
    if cand:
        d = np.load(cand[0])
        obs_img  = d["x"]
        z_hi_t   = d.get("z_hi_true")
        z_bye_t  = d.get("z_bye_true")
        z_sbi_t  = d.get("z_sbi_true")

    frames = []
    for gi, group in enumerate(groups):
        samples = [np.load(f) for f in group]
        def get(k): return np.array([s[k] for s in samples if k in s.files])
        z_hi  = get("z_hi.value")
        z_bye = get("z_bye.value")
        z_sbi = get("z_sbi.value")

        fig, axes = plt.subplots(2, 4, figsize=(14, 6))
        axes = axes.flatten()

        if obs_img is not None:
            axes[0].imshow(obs_img, cmap="gray")
            axes[0].set_title("Observed 64×64")
        axes[0].axis("off")

        def sc(ax, zv, i, j, zt, color, title):
            if len(zv):
                ax.scatter(zv[:, i], zv[:, j], s=5, alpha=0.5, c=color)
                if zt is not None:
                    ax.scatter([zt[i]], [zt[j]], s=80, marker="*", c="red", zorder=5)
            ax.set_xlim(-1.3, 1.3); ax.set_ylim(-1.3, 1.3)
            ax.set_title(title, fontsize=8); ax.set_aspect("equal")

        sc(axes[1], z_hi,  0, 1, z_hi_t,  "steelblue",   "HI: H pos")
        sc(axes[2], z_hi,  2, 3, z_hi_t,  "deepskyblue", "HI: I pos")
        sc(axes[3], z_bye, 0, 1, z_bye_t, "darkorange",  "BYE: B pos")
        sc(axes[4], z_bye, 2, 3, z_bye_t, "orange",      "BYE: Y pos")
        sc(axes[5], z_bye, 4, 5, z_bye_t, "coral",       "BYE: E pos")
        sc(axes[6], z_sbi, 0, 1, z_sbi_t, "mediumseagreen", "SBI: S pos")
        sc(axes[7], z_sbi, 2, 3, z_sbi_t, "limegreen",      "SBI: B+I pos")

        pct = int(100 * gi / max(1, len(groups) - 1))
        fig.suptitle(f"3-word composite (HI+BYE+SBI) — {pct}%", fontsize=11)
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
