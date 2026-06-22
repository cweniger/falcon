"""Animated GIF for 11_flow_words."""
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
    obs_img = z_hi_true = z_bye_true = None
    cand = list(data_dir.glob("obs.npz"))
    if cand:
        d = np.load(cand[0])
        obs_img = d["x"]
        z_hi_true  = d.get("z_hi_true")
        z_bye_true = d.get("z_bye_true")

    frames = []
    for gi, group in enumerate(groups):
        samples = [np.load(f) for f in group]
        z_hi  = np.array([s["z_hi.value"]  for s in samples if "z_hi.value"  in s.files])
        z_bye = np.array([s["z_bye.value"] for s in samples if "z_bye.value" in s.files])

        fig, axes = plt.subplots(1, 5, figsize=(16, 3.5))

        if obs_img is not None:
            axes[0].imshow(obs_img, cmap="gray"); axes[0].set_title("Observed 64×64 x")
        axes[0].axis("off")

        titles  = ["HI: H pos", "HI: I pos", "BYE: B pos", "BYE: Y+E pos"]
        colors  = ["steelblue", "deepskyblue", "darkorange", "coral"]
        for ai, (title, color) in enumerate(zip(titles, colors)):
            ax = axes[ai + 1]
            if ai < 2:
                zv = z_hi; i, j = 2*ai, 2*ai+1; zt = z_hi_true
            elif ai == 2:
                zv = z_bye; i, j = 0, 1; zt = z_bye_true
            else:
                # Overlay Y and E
                if len(z_bye):
                    ax.scatter(z_bye[:, 2], z_bye[:, 3], s=5, alpha=0.4, c="darkorange", label="Y")
                    ax.scatter(z_bye[:, 4], z_bye[:, 5], s=5, alpha=0.4, c="salmon", label="E")
                    ax.legend(fontsize=6, loc="upper right")
                    if z_bye_true is not None:
                        ax.scatter([z_bye_true[2], z_bye_true[4]],
                                   [z_bye_true[3], z_bye_true[5]], s=60, marker="*", c="red", zorder=5)
                ax.set_xlim(-1.3, 1.3); ax.set_ylim(-1.3, 1.3)
                ax.set_title(title, fontsize=8); ax.set_aspect("equal")
                continue
            if len(zv):
                ax.scatter(zv[:, i], zv[:, j], s=6, alpha=0.5, c=color)
                if zt is not None:
                    ax.scatter([zt[i]], [zt[j]], s=80, marker="*", c="red", zorder=5)
            ax.set_xlim(-1.3, 1.3); ax.set_ylim(-1.3, 1.3)
            ax.set_title(title, fontsize=8); ax.set_aspect("equal")

        pct = int(100 * gi / max(1, len(groups) - 1))
        fig.suptitle(f"NSF Flow — Training {pct}%  ({gi+1}/{len(groups)})", fontsize=10)
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
