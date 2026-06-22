"""Animated GIF for 12_four_components."""
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
    obs_img = z_hi_true = z_bye_true = z_dot_true = z_bg_true = None
    cand = list(data_dir.glob("obs.npz"))
    if cand:
        d = np.load(cand[0])
        obs_img     = d["x"]
        z_hi_true   = d.get("z_hi_true")
        z_bye_true  = d.get("z_bye_true")
        z_dot_true  = d.get("z_dot_true")
        z_bg_true   = d.get("z_bg_true")

    frames = []
    for gi, group in enumerate(groups):
        samples = [np.load(f) for f in group]

        def get(key):
            return np.array([s[key] for s in samples if key in s.files])

        z_hi  = get("z_hi.value")
        z_bye = get("z_bye.value")
        z_dot = get("z_dot.value")
        z_bg  = get("z_bg.value")

        fig, axes = plt.subplots(1, 6, figsize=(18, 3.2))

        if obs_img is not None:
            axes[0].imshow(obs_img, cmap="gray"); axes[0].set_title("Observed x")
        axes[0].axis("off")

        def scatter2d(ax, zv, i, j, zt, color, title):
            if len(zv):
                ax.scatter(zv[:, i], zv[:, j], s=5, alpha=0.5, c=color)
                if zt is not None:
                    ax.scatter([zt[i]], [zt[j]], s=80, marker="*", c="red", zorder=5)
            ax.set_xlim(-1.3, 1.3); ax.set_ylim(-1.3, 1.3)
            ax.set_title(title, fontsize=8); ax.set_aspect("equal")

        scatter2d(axes[1], z_hi,  0, 1, z_hi_true,  "steelblue",   "H pos (Flow)")
        scatter2d(axes[2], z_hi,  2, 3, z_hi_true,  "deepskyblue", "I pos (Flow)")
        scatter2d(axes[3], z_bye, 0, 1, z_bye_true, "darkorange",  "B pos (Flow)")
        scatter2d(axes[4], z_dot, 0, 1, z_dot_true, "green",       "Circle cx,cy")

        # BYE Y+E overlaid, and bg on same panel
        ax5 = axes[5]
        if len(z_bye):
            ax5.scatter(z_bye[:, 2], z_bye[:, 3], s=4, alpha=0.3, c="orange", label="Y")
            ax5.scatter(z_bye[:, 4], z_bye[:, 5], s=4, alpha=0.3, c="salmon", label="E")
            ax5.legend(fontsize=6)
            if z_bye_true is not None:
                ax5.scatter([z_bye_true[2], z_bye_true[4]],
                            [z_bye_true[3], z_bye_true[5]], s=60, marker="*", c="red", zorder=5)
        ax5.set_xlim(-1.3, 1.3); ax5.set_ylim(-1.3, 1.3)
        ax5.set_title("Y+E pos (Flow)", fontsize=8); ax5.set_aspect("equal")

        pct = int(100 * gi / max(1, len(groups) - 1))
        fig.suptitle(f"4-component inference — {pct}%", fontsize=10)
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
