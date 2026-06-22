"""
Generate an animated GIF from buffer snapshots.

Usage:  python make_gif.py output/run_01
The GIF shows how the inferred letter positions evolve during training,
with the observed image shown alongside each scatter frame.
"""
import sys, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import imageio.v2 as imageio
from pathlib import Path


def load_snapshots(snap_dir, every=1):
    """Load every `every`-th snapshot, return dict of lists."""
    files = sorted(Path(snap_dir).glob("*.npz"))
    files = files[::every]
    data = {}
    for f in files:
        d = np.load(f)
        for k in d.files:
            data.setdefault(k, []).append(d[k])
    return data, len(files)


def make_gif(run_dir, out_path=None, n_frames=30, fps=4):
    run_dir = Path(run_dir)
    snap_dir = run_dir / "buffer" / "snapshots"
    if not snap_dir.exists():
        print(f"No snapshots found in {snap_dir}")
        return

    files = sorted(snap_dir.glob("*.npz"))
    if not files:
        print("No snapshot files found.")
        return

    # Subsample to at most n_frames groups
    every = max(1, len(files) // n_frames)
    groups = [files[i:i+every] for i in range(0, len(files), every)][:n_frames]

    # Load observed image
    obs_candidates = list((run_dir.parent.parent / "data").glob("obs.npz"))
    obs_img = None
    if obs_candidates:
        obs_img = np.load(obs_candidates[0])["x"]

    # Load true z if available
    z_true = None
    if obs_candidates:
        d = np.load(obs_candidates[0])
        if "z_true" in d:
            z_true = d["z_true"]

    frames = []
    for gi, group in enumerate(groups):
        zvals = np.array([np.load(f)["z.value"] for f in group])  # (N, 4)

        fig = plt.figure(figsize=(8, 4))
        gs = gridspec.GridSpec(1, 3, width_ratios=[1.2, 1, 1])

        # Panel 1: observed image
        ax0 = fig.add_subplot(gs[0])
        if obs_img is not None:
            ax0.imshow(obs_img, cmap="gray", origin="upper")
            ax0.set_title("Observed x")
        ax0.axis("off")

        # Panel 2: H position scatter (x_H, y_H)
        ax1 = fig.add_subplot(gs[1])
        ax1.scatter(zvals[:, 0], zvals[:, 1], s=6, alpha=0.5, c="steelblue")
        if z_true is not None:
            ax1.scatter([z_true[0]], [z_true[1]], s=80, marker="*",
                        c="red", zorder=5, label="true")
            ax1.legend(fontsize=7)
        ax1.set_xlim(-1.2, 1.2); ax1.set_ylim(-1.2, 1.2)
        ax1.set_xlabel("x_H"); ax1.set_ylabel("y_H")
        ax1.set_title("Letter H position")
        ax1.set_aspect("equal")

        # Panel 3: I position scatter (x_I, y_I)
        ax2 = fig.add_subplot(gs[2])
        ax2.scatter(zvals[:, 2], zvals[:, 3], s=6, alpha=0.5, c="darkorange")
        if z_true is not None:
            ax2.scatter([z_true[2]], [z_true[3]], s=80, marker="*",
                        c="red", zorder=5, label="true")
            ax2.legend(fontsize=7)
        ax2.set_xlim(-1.2, 1.2); ax2.set_ylim(-1.2, 1.2)
        ax2.set_xlabel("x_I"); ax2.set_ylabel("y_I")
        ax2.set_title("Letter I position")
        ax2.set_aspect("equal")

        pct = int(100 * gi / max(1, len(groups) - 1))
        fig.suptitle(f"Training progress: {pct}%  (buffer snapshot group {gi+1}/{len(groups)})",
                     fontsize=10)
        fig.tight_layout()

        # Render to numpy
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)[..., :3]
        frames.append(buf)
        plt.close(fig)

    if out_path is None:
        out_path = run_dir / "training.gif"
    imageio.mimsave(str(out_path), frames, fps=fps, loop=0)
    print(f"Saved {len(frames)}-frame GIF to {out_path}")


if __name__ == "__main__":
    run_dir = sys.argv[1] if len(sys.argv) > 1 else "output/run"
    make_gif(run_dir)
