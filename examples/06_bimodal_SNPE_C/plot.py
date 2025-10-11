# -*- coding: utf-8 -*-
"""
Visualization script in the same style as example-10-high-dim:
- Reads posterior_10dim.joblib (list[dict] or dict[z])
- Reads x_obs_10dim.npy (and optionally true_z_10dim.npy / shift_10dim.npy)
- Constructs "theoretical posterior samples" (a dimension-wise mixture of z|x: mean x_obs ± 3σ, variance σ^2)
- Computes JS(network, theory)
- Draws a seaborn pairplot with per-dimensional histogram overlays
- Stores in npz: consistent with example storage fields (network_samples / true_samples / post_mu_1 / post_mu_2 / meta)
"""

import os
import numpy as np
import torch
import joblib
import matplotlib as mpl
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

import falcon
# from falcon.contrib.JSD import compute_js_divergence

# 1) 先加载你的风格
mpl.style.use("mystyle.mplstyle")

# 2) 适配 seaborn：只调字号，不覆写你 style 的其它设置
#    同时把想要的字号明确写到 rc（避免被 seaborn 覆盖）
sns.set_context(
    "talk",
    rc={
        "axes.labelsize": 22,  # 坐标轴标签
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 16,
        "figure.titlesize": 22,
    },
)

# 3) 覆盖“盒框”问题：关掉上/右脊梁；也顺便调大整图尺寸/导出清晰度
mpl.rcParams.update({
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.figsize": (14, 14),   # 整体画布更大
    "savefig.dpi": 200,
})

# 4) 如果你没装 LaTeX 或感觉慢，可临时关掉 usetex（你的 style 里是 True）
# mpl.rcParams["text.usetex"] = False

PATH_POSTERIOR = "samples_posterior.joblib"
PATH_X_OBS     = "data/x_obs_10dim.npy"
DIR_OUT        = os.path.dirname(PATH_POSTERIOR) or "."

# Optional truth value and shift (annotated if present)
PATH_TRUE_Z = os.path.join(os.path.dirname(PATH_X_OBS), "true_z_10dim.npy")
PATH_SHIFT  = os.path.join(os.path.dirname(PATH_X_OBS), "shift_10dim.npy")

# ===== Constant (same as in the example) =====
dim   = 10
SIGMA = 1e-2
MEAN_SHIFT = 3.0 * SIGMA
N_THEORY   = 1000
N_PLOT_MAX = 2000

def load_posterior_samples(path_joblib: str) -> torch.Tensor:
    data = joblib.load(path_joblib)
    if isinstance(data, list):
        assert len(data) > 0 and isinstance(data[0], dict) and "z" in data[0], \
            "joblib needs to be a list[dict] with a key 'z'"
        Z = np.stack([d["z"] for d in data], axis=0)  # (N,D)
    elif isinstance(data, dict):
        assert "z" in data, "joblib is a dict but does not contain the key 'z'"
        Z = np.asarray(data["z"])
    else:
        raise RuntimeError("Unknown posterior joblib storage format")

    assert Z.ndim == 2 and Z.shape[1] == dim, f"Z shape={Z.shape}，期望 (N,{dim})"
    return torch.from_numpy(Z).to(torch.float64)

def main():
    torch.set_default_dtype(torch.float64)

    # 1) network posterior samples
    Z_net = load_posterior_samples(PATH_POSTERIOR)  # (N,D)
    N_net = Z_net.shape[0]

    # Subsampling for drawing
    if N_net > N_PLOT_MAX:
        idx = torch.randperm(N_net)[:N_PLOT_MAX]
        Z_plot = Z_net[idx]
    else:
        Z_plot = Z_net

    # 2) x_obs
    x_obs = torch.from_numpy(np.load(PATH_X_OBS)).to(torch.float64).squeeze()
    assert x_obs.ndim == 1 and x_obs.numel() == dim, f"x_obs shape={tuple(x_obs.shape)}"

    # 3) Theoretical two peaks (mean): x_obs ± 3σ; covariance is σ^2 I
    post_mu_1 = x_obs - MEAN_SHIFT
    post_mu_2 = x_obs + MEAN_SHIFT

    # 4) Sampling from the theoretical posterior (selecting ± independently dimension by dimension, then adding N(0,σ^2))
    modes = torch.randint(0, 2, (N_THEORY, dim))
    mu_mat = torch.where(modes == 0, post_mu_1, post_mu_2).to(torch.float64)
    eps    = torch.randn(N_THEORY, dim) * SIGMA
    Z_theory = mu_mat + eps  # (N_THEORY,D)

    # # 5) calculate JS(network vs theory)
    # jsd = compute_js_divergence(Z_plot, Z_theory, dim)
    # print(f"JS Divergence (Network vs Theory): {jsd}")

    # 6) Read the true value and shift (add if present)
    true_z = None
    if os.path.exists(PATH_TRUE_Z):
        true_z = torch.from_numpy(np.load(PATH_TRUE_Z)).to(torch.float64).squeeze()
    shift = None
    if os.path.exists(PATH_SHIFT):
        shift = torch.from_numpy(np.load(PATH_SHIFT)).to(torch.float64).squeeze()

    # 7) Pairplot（Network vs Theory）
    theta_labels = [rf'$\theta_{{{i}}}$' for i in range(dim)]
    df_net = pd.DataFrame(Z_plot.numpy(),   columns=theta_labels)
    df_the = pd.DataFrame(Z_theory.numpy(), columns=theta_labels)
    df_net["Type"] = "Network"
    df_the["Type"] = "Theory"
    df_all = pd.concat([df_net, df_the], ignore_index=True)

    sns.set_context("talk")
    g = sns.pairplot(
        df_all,
        hue="Type",
        corner=True,
        height=1.8,
        plot_kws=dict(alpha=0.4, s=8),
        diag_kws=dict(alpha=0.6, common_norm=False),
    )

    for i in range(dim):
        ax = g.axes[i, i]
        if ax is not None:
            ax.axvline(post_mu_1[i].item(), linestyle="--", linewidth=1.2, color="black")
            ax.axvline(post_mu_2[i].item(), linestyle="--", linewidth=1.2, color="blue")

            if true_z is not None:
                ax.axvline(true_z[i].item(), linestyle="-.", linewidth=1.2, color="red")

    g.fig.subplots_adjust(top=0.93)
    g.fig.suptitle(f"Posterior Comparison (dim={dim}, sigma={SIGMA}, shift=±{MEAN_SHIFT})", y=0.995)

    os.makedirs(DIR_OUT, exist_ok=True)
    out_pair = os.path.join(DIR_OUT, "figures/posterior_pairplot.png")
    g.fig.savefig(out_pair, dpi=150, bbox_inches="tight")
    print(f"[OK] Saved: {out_pair}")
    g.fig.show()

    save_dir = os.path.join(DIR_OUT, "plot_data")
    os.makedirs(save_dir, exist_ok=True)
    data_path = os.path.join(save_dir, f"SNPE_A_multimodel_posteriors_dim{dim}_[-10k,10k]_0.5.npz")

    np.savez_compressed(
        data_path,
        network_samples=Z_plot.numpy(),     # (N_plot, D)
        true_samples=Z_theory.numpy(),      # (N_theory, D)
        post_mu_1=post_mu_1.numpy(),        # (D,)
        post_mu_2=post_mu_2.numpy(),        # (D,)
        true_z=(true_z.numpy() if true_z is not None else np.array([])),
        x_obs=x_obs.numpy(),                # (D,)
        dim=np.array(dim),
        sigma=np.array(SIGMA),
        rounds=np.array(1),
        prior_low=np.array(-10000.0),
        prior_high=np.array( 10000.0),
        # js_divergences=np.array([float(jsd)]),
    )
    print(f"[OK] Saved all plotting data to {data_path}")

if __name__ == "__main__":
    main()
