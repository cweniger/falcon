"""Animated GIF for 08_letters_noise — two-component inference."""
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

    # Load true params
    data_dir = run_dir.parent.parent / "data"
    z_pos_true = z_noise_true = obs_img = None
    cand = list(data_dir.glob("obs.npz"))
    if cand:
        d = np.load(cand[0])
        obs_img = d["x"]
        z_pos_true = d.get("z_pos_true")
        z_noise_true = d.get("z_noise_true")

    frames = []
    for gi, group in enumerate(groups):
        samples = [np.load(f) for f in group]

        fig, axes = plt.subplots(1, 4, figsize=(12, 3.5))

        # Observed image
        if obs_img is not None:
            axes[0].imshow(obs_img, cmap="gray", origin="upper")
        axes[0].set_title("Observed x"); axes[0].axis("off")

        # z_pos: H position
        zp = np.array([s["z_pos.value"] for s in samples if "z_pos.value" in s.files])
        if len(zp):
            axes[1].scatter(zp[:, 0], zp[:, 1], s=8, alpha=0.5, c="steelblue")
            if z_pos_true is not None:
                axes[1].scatter([z_pos_true[0]], [z_pos_true[1]], s=80, marker="*", c="red")
        axes[1].set_xlim(-1.3, 1.3); axes[1].set_ylim(-1.3, 1.3)
        axes[1].set_title("H position"); axes[1].set_aspect("equal")

        # z_pos: I position
        if len(zp) > 0:
            axes[2].scatter(zp[:, 2], zp[:, 3], s=8, alpha=0.5, c="darkorange")
            if z_pos_true is not None:
                axes[2].scatter([z_pos_true[2]], [z_pos_true[3]], s=80, marker="*", c="red")
        axes[2].set_xlim(-1.3, 1.3); axes[2].set_ylim(-1.3, 1.3)
        axes[2].set_title("I position"); axes[2].set_aspect("equal")

        # z_noise
        zn = np.array([s["z_noise.value"] for s in samples if "z_noise.value" in s.files])
        if len(zn):
            axes[3].scatter(zn[:, 0], zn[:, 1], s=8, alpha=0.5, c="purple")
            if z_noise_true is not None:
                axes[3].scatter([z_noise_true[0]], [z_noise_true[1]], s=80, marker="*", c="red")
        axes[3].set_xlim(-1.3, 1.3); axes[3].set_ylim(-1.3, 1.3)
        axes[3].set_xlabel("log_brightness"); axes[3].set_ylabel("log_noise")
        axes[3].set_title("Noise params"); axes[3].set_aspect("equal")

        pct = int(100 * gi / max(1, len(groups) - 1))
        fig.suptitle(f"Training {pct}%  (group {gi+1}/{len(groups)})", fontsize=10)
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
