# -*- coding: utf-8 -*-
import os
import numpy as np
import torch
import joblib
import matplotlib as mpl
import matplotlib.pyplot as plt
import corner
from matplotlib.ticker import FixedLocator, FixedFormatter
from matplotlib.patches import Patch

# ---------------- paths & constants ----------------
PATH_POSTERIOR     = "samples_posterior.joblib"
PATH_X_OBS         = "data/x_obs_10dim.npy"
PATH_TRAIN_STATS   = "plot_data/SNPE_C_multimodel_training_loss_and_theta_stats_[-10k,10k]_0.5.npz"

DIR_OUT        = os.path.dirname(PATH_POSTERIOR) or "."
DIM        = 10
SIGMA      = 1e-2
MEAN_SHIFT = 3.0 * SIGMA
N_THEORY   = 1000
N_PLOT_MAX = 4000

COLOR_REF = "#bc5090"   # 粉色（用于：Reference / Validation）
COLOR_NET = "#4C92C3"   # 蓝色（用于：Network   / Training）

mpl.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 300,
    "font.family": "serif", "mathtext.fontset": "cm",
    "font.size": 12, "axes.labelsize": 13, "axes.titlesize": 13,
    "xtick.labelsize": 11, "ytick.labelsize": 11,
    "axes.linewidth": 1.1,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.major.size": 4, "ytick.major.size": 4,
    "xtick.minor.size": 2, "ytick.minor.size": 2,
    "axes.spines.top": True, "axes.spines.right": True,
    "axes.formatter.useoffset": False, "axes.formatter.use_mathtext": True,
})

# ---------------- helpers ----------------
def load_posterior(path, dim):
    data = joblib.load(path)
    if isinstance(data, list):
        assert len(data) and isinstance(data[0], dict) and "z" in data[0]
        Z = np.stack([d["z"] for d in data], axis=0)
    elif isinstance(data, dict):
        assert "z" in data
        Z = np.asarray(data["z"])
    else:
        raise RuntimeError("posterior joblib 格式未知")
    assert Z.ndim == 2 and Z.shape[1] == dim
    return torch.as_tensor(Z, dtype=torch.float64)

def choose_global_scale(arrays, target_max=12.0):
    vmax = max(float(np.nanmax(np.abs(a))) for a in arrays)
    if vmax <= 0:
        return 0, 1.0
    k = int(np.floor(np.log10(vmax / target_max)))
    return k, 10.0**k

def set_two_ticks_4dp(ax, positions, axis="x"):
    labels = [f"{p:.4f}" for p in positions]
    if axis == "x":
        ax.xaxis.set_major_locator(FixedLocator(positions))
        ax.xaxis.set_major_formatter(FixedFormatter(labels))
    else:
        ax.yaxis.set_major_locator(FixedLocator(positions))
        ax.yaxis.set_major_formatter(FixedFormatter(labels))

def add_loss_axes_and_move_legend(fig, axes_grid, path_npz, handles):
    # 角图可见子图的包围盒
    poss = [ax.get_position() for row in axes_grid for ax in row if ax.get_visible()]
    xmin = min(p.x0 for p in poss)
    xmax = max(p.x1 for p in poss)
    ymax = max(p.y1 for p in poss)

    # 小图尺寸（较高）
    width, height = 0.24, 0.36
    gap_x = 0.02
    x0 = min(0.98 - width, xmax + gap_x)
    y0 = max(0.15, ymax - height)

    axL = fig.add_axes([x0, y0, width, height])

    if os.path.exists(path_npz):
        dat = np.load(path_npz)
        epochs     = dat["epochs"]
        loss_train = dat["loss_train_avg"]
        loss_val   = dat["loss_val_avg"]

        # 颜色按你的要求：蓝=Training，粉=Validation
        axL.plot(epochs, loss_train, color=COLOR_NET, lw=1.8, label="Training")
        axL.plot(epochs, loss_val,   color=COLOR_REF, lw=1.0, label="Validation")
    else:
        print(f"[warn] train stats not found: {path_npz} (skip curves)")

    axL.set_xlabel("Epoch")
    axL.set_ylabel("Train/Validation Loss")
    axL.tick_params(direction="in")
    for s in ("left", "right", "top", "bottom"):
        axL.spines[s].set_visible(True)
        axL.spines[s].set_linewidth(1.1)

    # 在小图内部加图例（右上角），与主图例无关
    axL.legend(loc="upper right", frameon=False, fontsize=12, handlelength=2.0)

    # 把主图例移到小图左侧，避免重叠
    legend_x = x0 - 0.26
    legend_y = y0 + height * 0.80
    fig.legend(handles=handles, frameon=False, fontsize=12,
               loc="center", bbox_to_anchor=(legend_x, legend_y))

# ---------------- main ----------------
def main():
    torch.set_default_dtype(torch.float64)

    # 网络样本
    Z_net_full = load_posterior(PATH_POSTERIOR, DIM)
    if len(Z_net_full) > N_PLOT_MAX:
        idx = torch.randperm(len(Z_net_full))[:N_PLOT_MAX]
        Z_net = Z_net_full[idx]
    else:
        Z_net = Z_net_full
    Z_net_np = Z_net.numpy()

    # 观测与理论双峰
    x_obs = torch.from_numpy(np.load(PATH_X_OBS)).to(torch.float64).squeeze()
    assert x_obs.ndim == 1 and x_obs.numel() == DIM
    mu1 = (x_obs - MEAN_SHIFT)
    mu2 = (x_obs + MEAN_SHIFT)

    modes  = torch.randint(0, 2, (N_THEORY, DIM))
    mu_mat = torch.where(modes == 0, mu1, mu2).to(torch.float64)
    Z_theory = (mu_mat + torch.randn(N_THEORY, DIM) * SIGMA).numpy()

    # 全局缩放
    k, scale = choose_global_scale([Z_net_np, Z_theory, mu1.numpy(), mu2.numpy()])
    s = 1.0 / scale
    Z_net_s    = Z_net_np * s
    Z_theory_s = Z_theory * s
    mu1_s, mu2_s = mu1.numpy() * s, mu2.numpy() * s

    # Corner：Reference→Network
    fig = corner.corner(
        Z_theory_s,
        color=COLOR_REF,
        plot_datapoints=True, markersize=1.8, alpha=0.22,
        fill_contours=False, plot_contours=True, levels=[0.5, 0.8, 0.95],
        contour_kwargs=dict(linewidths=1.2),
        bins=35, smooth=0.9,
        hist_kwargs=dict(density=True, histtype="step", linewidth=1.5),
        labels=None, max_n_ticks=2, use_math_text=True, quiet=True,
    )
    corner.corner(
        Z_net_s,
        color=COLOR_NET,
        plot_datapoints=True, markersize=1.8, alpha=0.22,
        fill_contours=False, plot_contours=True, levels=[0.5, 0.8, 0.95],
        contour_kwargs=dict(linewidths=1.2),
        bins=35, smooth=0.9,
        hist_kwargs=dict(density=True, histtype="step", linewidth=1.5),
        labels=None, max_n_ticks=2, use_math_text=True, fig=fig, quiet=True,
    )

    axes = np.array(fig.axes).reshape((DIM, DIM))
    names = [rf"$z_{{{i}}}$" for i in range(DIM)]

    # 修饰
    for i in range(DIM):
        for j in range(DIM):
            ax = axes[i, j]
            if i < j:
                ax.set_visible(False)
                continue

            for sname in ("left", "right", "top", "bottom"):
                ax.spines[sname].set_visible(True)
                ax.spines[sname].set_linewidth(1.1)
            ax.tick_params(which="both", direction="in", pad=2)

            show_x = (i == DIM - 1)
            show_y = (j == 0)

            if show_x:
                set_two_ticks_4dp(ax, [mu1_s[j], mu2_s[j]], axis="x")
            else:
                ax.set_xticks([]); ax.set_xticklabels([])

            if show_y and (i != j):
                set_two_ticks_4dp(ax, [mu1_s[i], mu2_s[i]], axis="y")
            else:
                ax.set_yticks([]); ax.set_yticklabels([])

            if i == j:
                ax.text(0.5, 1.10, names[i],
                        transform=ax.transAxes, ha="center", va="bottom", fontsize=13)
                ax.axvline(mu1_s[i], ls="--", lw=1.0, c="k", alpha=0.85, zorder=5)
                ax.axvline(mu2_s[i], ls="--", lw=1.0, c="k", alpha=0.85, zorder=5)
            elif i > j:
                ax.plot([mu1_s[j], mu2_s[j], mu1_s[j], mu2_s[j]],
                        [mu1_s[i], mu1_s[i], mu2_s[i], mu2_s[i]],
                        "o", ms=3.2, mec="k", mfc="k", zorder=6, alpha=0.95)

    # 主图例句柄（角图）
    handles = [
        Patch(edgecolor=COLOR_REF, facecolor="none", label="Reference Samples", lw=1.8),
        Patch(edgecolor=COLOR_NET, facecolor="none", label="Learned Posterior", lw=1.8),
    ]

    # 左上角全局缩放标注
    if k != 0:
        fig.text(0.03, 0.965, rf"$\times 10^{{-{k}}}$",
                 ha="left", va="top", fontsize=12)

    # 布局与尺寸
    fig.subplots_adjust(top=0.94, left=0.08, right=0.98, bottom=0.07,
                        hspace=0.09, wspace=0.09)
    base = 1.9
    fig.set_size_inches(base*DIM*0.6, base*DIM*0.6)

    # 添加损失曲线 + 移动主图例
    add_loss_axes_and_move_legend(fig, axes, PATH_TRAIN_STATS, handles)

    # 保存
    out_dir = os.path.join(DIR_OUT, "figures")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "posterior_corner_with_loss.png")
    fig.savefig(out_path, bbox_inches="tight")
    print(f"[OK] Saved: {out_path}")

if __name__ == "__main__":
    main()
