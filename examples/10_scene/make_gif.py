"""Animated GIF for 10_scene — three-component inference."""
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
    obs_img = z_word_true = z_dot_true = z_noise_true = None
    cand = list(data_dir.glob("obs.npz"))
    if cand:
        d = np.load(cand[0])
        obs_img = d["x"]
        z_word_true  = d.get("z_word_true")
        z_dot_true   = d.get("z_dot_true")
        z_noise_true = d.get("z_noise_true")

    frames = []
    for gi, group in enumerate(groups):
        samples = [np.load(f) for f in group]
        z_word  = np.array([s["z_word.value"]  for s in samples if "z_word.value"  in s.files])
        z_dot   = np.array([s["z_dot.value"]   for s in samples if "z_dot.value"   in s.files])
        z_noise = np.array([s["z_noise.value"] for s in samples if "z_noise.value" in s.files])

        fig, axes = plt.subplots(1, 5, figsize=(15, 3.2))

        if obs_img is not None:
            axes[0].imshow(obs_img, cmap="gray")
        axes[0].set_title("Observed x"); axes[0].axis("off")

        # H position
        if len(z_word):
            axes[1].scatter(z_word[:, 0], z_word[:, 1], s=6, alpha=0.5, c="steelblue")
            if z_word_true is not None:
                axes[1].scatter([z_word_true[0]], [z_word_true[1]], s=80, marker="*", c="red")
        axes[1].set_xlim(-1.3, 1.3); axes[1].set_ylim(-1.3, 1.3)
        axes[1].set_title("Word: H pos"); axes[1].set_aspect("equal")

        # I position
        if len(z_word):
            axes[2].scatter(z_word[:, 2], z_word[:, 3], s=6, alpha=0.5, c="deepskyblue")
            if z_word_true is not None:
                axes[2].scatter([z_word_true[2]], [z_word_true[3]], s=80, marker="*", c="red")
        axes[2].set_xlim(-1.3, 1.3); axes[2].set_ylim(-1.3, 1.3)
        axes[2].set_title("Word: I pos"); axes[2].set_aspect("equal")

        # Circle center
        if len(z_dot):
            axes[3].scatter(z_dot[:, 0], z_dot[:, 1], s=6, alpha=0.5, c="darkorange")
            if z_dot_true is not None:
                axes[3].scatter([z_dot_true[0]], [z_dot_true[1]], s=80, marker="*", c="red")
        axes[3].set_xlim(-1.3, 1.3); axes[3].set_ylim(-1.3, 1.3)
        axes[3].set_title("Circle center"); axes[3].set_aspect("equal")

        # Background vs circle radius
        if len(z_dot) and len(z_noise):
            n = min(len(z_dot), len(z_noise))
            axes[4].scatter(z_noise[:n, 0], z_dot[:n, 2], s=6, alpha=0.5, c="purple")
            if z_noise_true is not None and z_dot_true is not None:
                axes[4].scatter([z_noise_true[0]], [z_dot_true[2]], s=80, marker="*", c="red")
        axes[4].set_xlim(-1.3, 1.3); axes[4].set_ylim(-1.3, 1.3)
        axes[4].set_xlabel("log_bg"); axes[4].set_ylabel("circle_r")
        axes[4].set_title("Bg vs radius"); axes[4].set_aspect("equal")

        pct = int(100 * gi / max(1, len(groups) - 1))
        fig.suptitle(f"Training {pct}%", fontsize=10)
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
