# %% [markdown]
# # DynamicSVD — streaming data compression for simulation-based inference
#
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cweniger/falcon/blob/main/notebooks/dynamic_svd.ipynb)
#
# `falcon.embeddings.DynamicSVD` is a streaming SVD embedding with
# Procrustes-stabilized output, Wiener shrinkage, and optional noise
# whitening. This notebook demonstrates each capability on an idealized
# setup: $n = 100$ data bins, 4-parameter models, and **exact
# Fisher-information ground truth** — no falcon runs and no neural-network
# training involved.
#
# 1. **Basics** — why ranking directions by variance finds the informative
#    subspace (linear model, white noise).
# 2. **How many components?** — non-linear models, prior width, and the
#    Wiener shrinkage that makes over-provisioning cheap.
# 3. **Streaming & Procrustes** — why a stable output frame matters when
#    the proposal distribution narrows during adaptive inference.
# 4. **Colored noise** — how correlated noise breaks variance ranking, and
#    how the Toeplitz whitener restores it.
#
# *This notebook is paired with `dynamic_svd.py` via Jupytext; the `.py`
# file is the version-controlled source of truth.*

# %%
# On Colab, install falcon from GitHub (torch and numpy are preinstalled).
import importlib.util

if importlib.util.find_spec("falcon") is None:
    import subprocess, sys
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q",
         "git+https://github.com/cweniger/falcon.git"],
        check=True,
    )

# %%
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from falcon.embeddings import DynamicSVD, ToeplitzWhitener, hartley_transform

torch.set_default_dtype(torch.float64)
torch.manual_seed(0)

N = 100                      # number of data bins
P = 4                        # number of model parameters
t = torch.linspace(0.0, 1.0, N)

# %% [markdown]
# ## Fisher-information toolbox
#
# For a model $f(\theta)$ with Gaussian noise of covariance $C$, the Fisher
# information matrix is exact and analytic,
#
# $$ F(\theta) = J(\theta)^\top C^{-1} J(\theta), \qquad
#    J = \partial f / \partial \theta , $$
#
# with $J$ obtained from autograd — valid for linear *and* non-linear
# models. A trained `DynamicSVD` in eval mode is a fixed linear map
# $z = A x$ (plus an irrelevant offset), so the Fisher information of the
# *embedded* data is
#
# $$ F_{\rm emb} = (AJ)^\top \, (A C A^\top)^{-1} \, (AJ) . $$
#
# Comparing $F_{\rm emb}$ against $F$ tells us exactly how much information
# the compression $x \mapsto z$ discards. As a scalar figure of merit we use
#
# $$ r = \left( \det F_{\rm emb} / \det F \right)^{1/P} \in [0, 1] , $$
#
# which reads as an average error-bar deflation per parameter ($r = 1$:
# lossless compression).
#
# Note that $F_{\rm emb}$ is invariant under any invertible reparametrization
# of $z$ — rotations (Procrustes) and per-component rescalings (shrinkage,
# normalization) change the *conditioning* of the embedding output, never its
# *information content*. We will use this fact repeatedly.


# %%
def fisher(model, theta, C=None):
    """Exact Fisher matrix J^T C^-1 J via autograd (C=None means C=I)."""
    J = torch.autograd.functional.jacobian(model, theta)
    if C is None:
        return J.T @ J
    return J.T @ torch.linalg.solve(C, J)


def embedding_matrix(emb):
    """Extract the linear map A of a trained embedding (eval mode)."""
    emb.eval()
    return torch.autograd.functional.jacobian(
        lambda x: emb(x.unsqueeze(0)).squeeze(0), torch.zeros(N)
    )


def fisher_embedded(emb, model, theta, C=None):
    """Fisher matrix of the embedded data z = A x."""
    A = embedding_matrix(emb)
    J = torch.autograd.functional.jacobian(model, theta)
    C = torch.eye(N) if C is None else C
    AJ = A @ J
    S = A @ C @ A.T + 1e-12 * torch.eye(A.shape[0])
    return AJ.T @ torch.linalg.solve(S, AJ)


def info_ratio(F_emb, F_full):
    """(det F_emb / det F_full)^(1/P): 1 = lossless, 0 = unconstrained."""
    p = F_full.shape[0]
    jitter = 1e-12 * torch.eye(p)
    ld_emb = torch.linalg.slogdet(F_emb + jitter).logabsdet
    ld_full = torch.linalg.slogdet(F_full + jitter).logabsdet
    return float(torch.exp((ld_emb - ld_full) / p))


def train_dynamic_svd(emb, simulate, n_updates=40, batch_size=64):
    """Stream batches of simulations through the embedding."""
    emb.train()
    for _ in range(n_updates):
        x, signal = simulate(batch_size)
        emb.update(x, signal=signal)
    emb.eval()
    return emb


# %%
def _add_ellipse(ax, mean, cov2, color, nsig, lw, alpha):
    vals, vecs = np.linalg.eigh(cov2)
    angle = np.degrees(np.arctan2(vecs[1, 1], vecs[0, 1]))
    ax.add_patch(
        Ellipse(
            mean,
            2 * nsig * np.sqrt(vals[1]),
            2 * nsig * np.sqrt(vals[0]),
            angle=angle,
            fill=False,
            edgecolor=color,
            lw=lw,
            alpha=alpha,
        )
    )


def plot_fisher_corner(fishers, labels, colors, theta0, names, title=None):
    """Corner plot of 1- and 2-sigma Fisher ellipses (covariance = F^-1)."""
    p = len(theta0)
    covs = [torch.linalg.inv(F).numpy() for F in fishers]
    center = theta0.numpy()
    stds = np.array([np.sqrt(np.diag(c)) for c in covs]).max(axis=0)
    fig, axes = plt.subplots(p, p, figsize=(2.0 * p, 2.0 * p), constrained_layout=True)
    for i in range(p):
        for j in range(p):
            ax = axes[i, j]
            if j > i:
                ax.set_visible(False)
                continue
            lo_j, hi_j = center[j] - 3.5 * stds[j], center[j] + 3.5 * stds[j]
            if i == j:
                xs = np.linspace(lo_j, hi_j, 200)
                for c, col, lab in zip(covs, colors, labels):
                    s = np.sqrt(c[i, i])
                    ax.plot(xs, np.exp(-0.5 * ((xs - center[i]) / s) ** 2),
                            color=col, label=lab)
                ax.set_yticks([])
            else:
                for c, col in zip(covs, colors):
                    sub = c[np.ix_([j, i], [j, i])]
                    _add_ellipse(ax, (center[j], center[i]), sub, col, 1, 1.5, 1.0)
                    _add_ellipse(ax, (center[j], center[i]), sub, col, 2, 1.0, 0.5)
                ax.set_ylim(center[i] - 3.5 * stds[i], center[i] + 3.5 * stds[i])
            ax.set_xlim(lo_j, hi_j)
            if i == p - 1:
                ax.set_xlabel(names[j])
            else:
                ax.set_xticklabels([])
            if j == 0 and i > 0:
                ax.set_ylabel(names[i])
            else:
                ax.set_yticklabels([])
    handles, labs = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labs, loc="upper right", frameon=False)
    if title:
        fig.suptitle(title)
    return fig


# %% [markdown]
# ## 1. Basics: linear model, white noise
#
# Data model: $x = W\theta + n$ with $\theta \sim \mathcal N(0, \mathbb 1_4)$
# and unit white noise $n \sim \mathcal N(0, \mathbb 1_{100})$. The design
# matrix $W$ has four columns: a constant, a slope, and two sinusoids.
#
# **Why does an SVD of raw data find the informative directions?** The data
# covariance is
#
# $$ C_x = W \,\Sigma_\theta\, W^\top + \sigma^2 \mathbb 1 , $$
#
# i.e. the identity plus a rank-4 term whose eigenvectors span the column
# space of $W$. Ranking directions by variance therefore puts the entire
# signal subspace into the top four components, and projecting onto them
# leaves the Fisher matrix $F = W^\top W / \sigma^2$ untouched. `DynamicSVD`
# estimates exactly this eigenbasis, but *online*, from streamed batches.

# %%
torch.manual_seed(1)

W = torch.stack(
    [
        torch.ones(N),
        t - 0.5,
        torch.sin(2 * np.pi * 3 * t),
        torch.cos(2 * np.pi * 7 * t),
    ],
    dim=1,
)  # (N, P)

theta0_lin = torch.tensor([0.5, -0.3, 0.8, 0.4])
names_lin = [r"$\theta_1$", r"$\theta_2$", r"$\theta_3$", r"$\theta_4$"]


def linear_model(theta):
    return W @ theta


def simulate_linear(m):
    theta = torch.randn(m, P)
    signal = theta @ W.T
    return signal + torch.randn(m, N), signal


fig, axes = plt.subplots(1, 2, figsize=(9, 2.8), constrained_layout=True)
for k in range(P):
    axes[0].plot(t, W[:, k], label=names_lin[k])
axes[0].legend(fontsize=8, ncol=2)
axes[0].set_title("design matrix columns")
x_demo, f_demo = simulate_linear(1)
axes[1].plot(t, x_demo[0], color="0.6", lw=0.8, label="data $x$")
axes[1].plot(t, f_demo[0], color="k", label="signal $W\\theta$")
axes[1].legend(fontsize=8)
axes[1].set_title("one realization")
for ax in axes:
    ax.set_xlabel("$t$")
plt.show()

# %%
emb_lin = train_dynamic_svd(
    DynamicSVD(n_components=8, buffer_size=64), simulate_linear
)

F_true_lin = fisher(linear_model, theta0_lin)
F_emb_lin = fisher_embedded(emb_lin, linear_model, theta0_lin)

print(f"information ratio (8 of 100 dims kept): r = {info_ratio(F_emb_lin, F_true_lin):.4f}")

plot_fisher_corner(
    [F_true_lin, F_emb_lin],
    ["full data (analytic)", "DynamicSVD embedding"],
    ["k", "C0"],
    theta0_lin,
    names_lin,
    title="Linear model, white noise: compression is lossless",
)
plt.show()

# %% [markdown]
# The embedded Fisher ellipses lie on top of the analytic full-data ones and
# the information ratio is $r \approx 1$: compressing 100 bins to 8
# streaming-SVD coefficients loses essentially nothing, because the signal
# subspace has rank 4 and sits well above the unit noise floor.

# %% [markdown]
# ## 2. How many components? Non-linear models and Wiener shrinkage
#
# Real simulators are non-linear. We use a Gaussian bump on a baseline,
#
# $$ f(t;\theta) = A \exp\!\left( -\tfrac{(t - t_0)^2}{2 w^2} \right) + b ,
#    \qquad \theta = (A,\, t_0,\, w,\, b) , $$
#
# with unit white noise. Non-linearity changes the picture in one essential
# way: the set $\{ f(\theta) \}$ is now a *curved* manifold. Its tangent
# directions — the locally informative ones — rotate as $\theta$ varies, so
# a single rank-4 basis is no longer automatically enough. How many
# components we need depends on how much of the manifold the prior explores.

# %%
torch.manual_seed(2)

theta0_bump = torch.tensor([2.0, 0.5, 0.08, 0.5])
names_bump = ["$A$", "$t_0$", "$w$", "$b$"]
delta_base = torch.tensor([1.5, 0.35, 0.05, 0.5])  # prior half-width at s = 1


def bump_model(theta):
    A, t0, w, b = theta
    return A * torch.exp(-0.5 * ((t - t0) / w) ** 2) + b


def bump_batch(theta):
    A, t0, w, b = (theta[:, k : k + 1] for k in range(P))
    return A * torch.exp(-0.5 * ((t[None, :] - t0) / w) ** 2) + b


def make_simulator(s, sigma=1.0):
    """Uniform prior of relative width s around the fiducial point."""

    def simulate(m):
        theta = theta0_bump + (2 * torch.rand(m, P) - 1) * (s * delta_base)
        signal = bump_batch(theta)
        return signal + sigma * torch.randn(m, N), signal

    return simulate


fig, axes = plt.subplots(1, 2, figsize=(9, 2.8), sharey=True, constrained_layout=True)
for ax, s in zip(axes, [0.1, 0.8]):
    _, sig = make_simulator(s)(20)
    ax.plot(t, sig.T, color="C0", lw=0.7, alpha=0.5)
    ax.plot(t, bump_model(theta0_bump), color="k", lw=1.5)
    ax.set_title(f"prior width s = {s}")
    ax.set_xlabel("$t$")
plt.show()

# %% [markdown]
# ### Information ratio vs. number of components
#
# We train a fresh `DynamicSVD` for every combination of retained components
# $k \in \{2, 4, 8, 16, 32\}$ and prior width
# $s \in \{0.05, 0.1, 0.2, 0.4, 0.8\}$, and evaluate the information ratio at
# the central fiducial point.

# %%
ks = [2, 4, 8, 16, 32]
widths = [0.05, 0.1, 0.2, 0.4, 0.8]

F_true_bump = fisher(bump_model, theta0_bump)

emb_sweep = {}
ratio_sweep = {}
for s in widths:
    simulate = make_simulator(s)
    for k in ks:
        emb = train_dynamic_svd(
            DynamicSVD(n_components=k, buffer_size=64), simulate
        )
        emb_sweep[(s, k)] = emb
        ratio_sweep[(s, k)] = info_ratio(
            fisher_embedded(emb, bump_model, theta0_bump), F_true_bump
        )

fig, ax = plt.subplots(figsize=(5.5, 3.5), constrained_layout=True)
for i, s in enumerate(widths):
    ax.plot(ks, [ratio_sweep[(s, k)] for k in ks], "o-",
            color=plt.cm.viridis(i / (len(widths) - 1)), label=f"s = {s}")
ax.set_xscale("log", base=2)
ax.set_xticks(ks, [str(k) for k in ks])
ax.axhline(1.0, color="0.7", lw=0.8, zorder=0)
ax.set_xlabel("retained components $k$")
ax.set_ylabel("information ratio $r$")
ax.legend(title="prior width", fontsize=8)
ax.set_title("Wider priors need more components")
plt.show()

# %% [markdown]
# Two things to read off:
#
# - For a **narrow prior** the curve saturates as soon as $k \geq 4$: locally
#   the manifold looks like its tangent plane, and rank-$P$ suffices — just
#   like the linear case.
# - For a **wide prior** saturation moves to larger $k$: the basis must cover
#   the manifold's curvature (bump positions all over the grid), not just one
#   tangent plane.
#
# This is a static picture of what happens *dynamically* in falcon's adaptive
# rounds, where the proposal distribution starts prior-wide and shrinks
# toward the posterior: early rounds effectively need a larger $k$ than late
# rounds.
#
# The evaluation above is at the central fiducial. To check the basis is
# adequate *everywhere* the prior reaches — the property that makes any
# downstream (also non-Gaussian) posterior safe — we repeat it at several
# random fiducial points of the widest prior:

# %%
torch.manual_seed(3)

s_wide = 0.8
fiducials = theta0_bump + (2 * torch.rand(5, P) - 1) * (s_wide * delta_base)

fig, ax = plt.subplots(figsize=(5.5, 3.5), constrained_layout=True)
for th in fiducials:
    F_th = fisher(bump_model, th)
    r_th = [
        info_ratio(fisher_embedded(emb_sweep[(s_wide, k)], bump_model, th), F_th)
        for k in ks
    ]
    ax.plot(ks, r_th, "-", color="C0", lw=0.8, alpha=0.6)
r_central = [ratio_sweep[(s_wide, k)] for k in ks]
ax.plot(ks, r_central, "o-", color="k", label="central fiducial")
ax.set_xscale("log", base=2)
ax.set_xticks(ks, [str(k) for k in ks])
ax.axhline(1.0, color="0.7", lw=0.8, zorder=0)
ax.set_xlabel("retained components $k$")
ax.set_ylabel("information ratio $r$")
ax.legend(fontsize=8)
ax.set_title(f"s = {s_wide}: adequacy across the prior (thin: random fiducials)")
plt.show()

# %% [markdown]
# ### The eigenvalue spectrum and the noise floor
#
# In units where the noise is white with unit variance, every direction
# receives a noise contribution of $1$: informative components sit *above*
# the floor $\lambda \approx 1$, noise-dominated ones sit *on* it. The scree
# plot is therefore the direct visual answer to "how many components carry
# signal here":

# %%
fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), constrained_layout=True)
for i, s in enumerate([0.1, 0.4, 0.8]):
    lam = emb_sweep[(s, 32)].eigenvalues
    color = plt.cm.viridis(i / 2)
    axes[0].semilogy(np.arange(1, 33), lam, "o-", ms=3, color=color, label=f"s = {s}")
    axes[1].plot(np.arange(1, 33), lam / (lam + 1.0), "o-", ms=3, color=color)
axes[0].axhline(1.0, color="0.5", ls="--", lw=0.8)
axes[0].text(20, 1.15, "noise floor", color="0.4", fontsize=8)
axes[0].set_xlabel("component index")
axes[0].set_ylabel(r"eigenvalue $\lambda$")
axes[0].legend(fontsize=8)
axes[0].set_title("scree plot (whitened units)")
axes[1].set_xlabel("component index")
axes[1].set_ylabel(r"shrinkage factor $\lambda/(\lambda+1)$")
axes[1].set_title("Wiener shrinkage weight")
plt.show()

# %% [markdown]
# ### Wiener shrinkage: over-provisioning is nearly free
#
# `DynamicSVD.forward` applies two per-component rescalings after projection:
# a Wiener factor $\lambda/(\lambda+1)$ (`shrinkage=True`) and a
# normalization $1/\sqrt{\lambda}$ to roughly unit output variance.
#
# Remember from the toolbox section: both are invertible rescalings, so the
# **Fisher information is exactly identical with shrinkage on or off** — we
# will not show Fisher plots here because the curves would coincide. What
# shrinkage changes is what the downstream network *sees*. Normalization
# alone boosts noise-dominated components to unit variance, making them
# indistinguishable from informative ones; with shrinkage they are damped, so
# the embedding self-selects its effective dimensionality:

# %%
emb_demo = emb_sweep[(0.4, 32)]
x_demo, _ = make_simulator(0.4)(2000)

# forward() output: shrinkage + normalization (as the downstream network sees it)
c_shrunk = emb_demo(x_demo)

# normalization only, computed manually (the shrinkage flag also controls
# normalization in the current implementation; see cweniger/falcon#113)
lam = emb_demo.eigenvalues
c_norm = ((x_demo @ emb_demo.components.T) / torch.sqrt(lam)) @ emb_demo._R.T

fig, ax = plt.subplots(figsize=(6.5, 3.2), constrained_layout=True)
idx = np.arange(1, 33)
ax.bar(idx - 0.2, c_norm.std(dim=0), width=0.4, label="normalization only", color="0.6")
ax.bar(idx + 0.2, c_shrunk.std(dim=0), width=0.4, label="with Wiener shrinkage", color="C0")
ax.set_xlabel("component index")
ax.set_ylabel("output std")
ax.legend(fontsize=8)
ax.set_title("Shrinkage damps noise-floor components; normalization alone hides them")
plt.show()

# %% [markdown]
# The same weights power `reconstruct()`, which maps data back to bin space
# through the shrunk projection — i.e. a Wiener denoiser:

# %%
torch.manual_seed(4)
x_one, f_one = make_simulator(0.4)(1)

fig, ax = plt.subplots(figsize=(6.5, 3.0), constrained_layout=True)
ax.plot(t, x_one[0], color="0.7", lw=0.8, label="noisy data")
ax.plot(t, f_one[0], color="k", label="true signal")
ax.plot(t, emb_demo.reconstruct(x_one)[0], color="C3", label="Wiener reconstruction")
ax.set_xlabel("$t$")
ax.legend(fontsize=8)
ax.set_title("reconstruct(): noise-floor components are suppressed")
plt.show()

# %% [markdown]
# **Practical guidance:** choose $k$ generously. The information-ratio curve
# gives the *minimum*; shrinkage makes the excess nearly free, because
# surplus components arrive at the network already damped toward zero.
#
# ### Caveat: the shrinkage formula assumes a unit noise floor
#
# The factor $\lambda/(\lambda+1)$ hard-codes the noise floor at
# $\lambda = 1$. That is guaranteed when a whitener is attached (Section 4)
# — and happens to hold above because we simulated $\sigma = 1$ noise — but
# in raw data units it silently fails. With $\sigma = 10$ noise, every
# eigenvalue sits far above 1 and the shrinkage factor is $\approx 1$ for
# *all* components, informative or not:

# %%
emb_sigma10 = train_dynamic_svd(
    DynamicSVD(n_components=32, buffer_size=64), make_simulator(0.4, sigma=10.0)
)

fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), constrained_layout=True)
for emb, lab, color in [(emb_demo, r"$\sigma = 1$", "C0"),
                        (emb_sigma10, r"$\sigma = 10$", "C3")]:
    lam = emb.eigenvalues
    axes[0].semilogy(np.arange(1, 33), lam, "o-", ms=3, color=color, label=lab)
    axes[1].plot(np.arange(1, 33), lam / (lam + 1.0), "o-", ms=3, color=color)
axes[0].axhline(1.0, color="0.5", ls="--", lw=0.8)
axes[0].axhline(100.0, color="C3", ls=":", lw=0.8)
axes[0].text(2, 130, r"floor at $\sigma^2 = 100$", color="C3", fontsize=8)
axes[0].set_xlabel("component index")
axes[0].set_ylabel(r"eigenvalue $\lambda$")
axes[0].legend(fontsize=8)
axes[1].set_xlabel("component index")
axes[1].set_ylabel(r"shrinkage factor $\lambda/(\lambda+1)$")
axes[1].set_ylim(0, 1.05)
axes[1].set_title(r"$\sigma = 10$: shrinkage silently does nothing")
plt.show()

# %% [markdown]
# The $\sigma = 10$ shrinkage curve is flat at $\approx 1$: no differential
# suppression at all. A self-calibrating floor,
# $\lambda / (\lambda + \lambda_{\rm floor})$ with $\lambda_{\rm floor}$
# estimated from the variance per discarded direction, fixes this and is
# tracked in [cweniger/falcon#113](https://github.com/cweniger/falcon/issues/113).
# Until then: either attach a whitener (Section 4) or make sure your data is
# scaled to unit noise variance.

# %% [markdown]
# ## 3. Streaming and Procrustes stabilization
#
# In falcon's adaptive inference the proposal distribution *narrows over
# time*, so the embedding is trained on a non-stationary stream. The SVD
# basis then evolves — and a basis ordered by eigenvalue *index* is
# treacherous: whenever two eigenvalues cross, components swap places, and
# each SVD update can flip signs at will. For the downstream network this
# means input $j$ suddenly measures something different than a moment ago.
#
# `DynamicSVD` therefore maintains a rotation $R$ (updated via orthogonal
# Procrustes alignment after every SVD update) that maps the current basis
# back to a *stable output frame*. Crucially, $R$ is orthogonal: it changes
# the parameterization of the retained subspace, not the subspace itself —
# the Fisher information is identical with or without it.
#
# We mimic adaptive inference by shrinking the prior width geometrically from
# $s = 0.8$ to $s = 0.02$ over 40 SVD updates, and record both views of the
# basis after each update: the raw index-ordered components $V_t$ and the
# stabilized frame $B_t = R_t V_t$.

# %%
torch.manual_seed(5)

K3 = 6
sweep = np.geomspace(0.8, 0.02, 40)

emb_stream = DynamicSVD(n_components=K3, buffer_size=64)
emb_stream.train()
Vs, Bs = [], []
for s in sweep:
    x, signal = make_simulator(float(s))(64)
    emb_stream.update(x)
    Vs.append(emb_stream.components.clone())
    Bs.append((emb_stream._R @ emb_stream.components).clone())
emb_stream.eval()

fig, axes = plt.subplots(1, 2, figsize=(10, 5.5), sharey=True, constrained_layout=True)
cmap = plt.cm.viridis
for ax, frames, title in [
    (axes[0], Vs, "raw components $V_t$ (index-ordered)"),
    (axes[1], Bs, "stabilized frame $B_t = R_t V_t$"),
]:
    for step, Vt in enumerate(frames):
        for i in range(K3):
            ax.plot(t, Vt[i] + 0.8 * i, color=cmap(step / (len(frames) - 1)),
                    lw=0.7, alpha=0.6)
    ax.set_yticks(0.8 * np.arange(K3), [f"comp {i+1}" for i in range(K3)])
    ax.set_xlabel("$t$")
    ax.set_title(title)
fig.colorbar(plt.cm.ScalarMappable(cmap=cmap), ax=axes, shrink=0.6,
             label="update step (prior narrowing)")
plt.show()

# %% [markdown]
# Left: as the prior narrows, eigenvalues cross and the index-ordered
# components swap and sign-flip between updates. Right: in the stabilized
# frame the same subspace deforms smoothly. A quantitative view — the largest
# change of any basis vector between consecutive updates (a sign flip gives
# $\|\Delta v\| = 2$):

# %%
def max_step_change(frames):
    return [
        (frames[i] - frames[i - 1]).norm(dim=1).max().item()
        for i in range(1, len(frames))
    ]


fig, ax = plt.subplots(figsize=(6.5, 3.2), constrained_layout=True)
ax.semilogy(max_step_change(Vs), "o-", ms=3, color="C3", label="raw $V_t$")
ax.semilogy(max_step_change(Bs), "o-", ms=3, color="C0", label="stabilized $B_t$")
ax.axhline(2.0, color="0.6", ls=":", lw=0.8)
ax.text(1, 1.6, "sign flip", color="0.5", fontsize=8)
ax.set_xlabel("update step")
ax.set_ylabel(r"$\max_i \, \| v_{i,t} - v_{i,t-1} \|$")
ax.legend(fontsize=8)
ax.set_title("Basis jumps between consecutive SVD updates")
plt.show()

# %% [markdown]
# Finally, the quantity the downstream network actually consumes: the
# embedding coefficients of one *fixed* observation, traced across the
# narrowing sequence. Without stabilization they jump discontinuously at
# every crossing; in the stable frame they drift smoothly:

# %%
torch.manual_seed(6)
x_fixed, _ = make_simulator(0.1)(1)
x_fixed = x_fixed[0]

c_raw = torch.stack([V @ x_fixed for V in Vs])   # (steps, K3)
c_stable = torch.stack([B @ x_fixed for B in Bs])

fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True, constrained_layout=True)
for i in range(3):
    axes[0].plot(c_raw[:, i], "o-", ms=2, lw=0.8, label=f"comp {i+1}")
    axes[1].plot(c_stable[:, i], "o-", ms=2, lw=0.8)
axes[0].set_title("raw projection")
axes[1].set_title("stabilized frame")
axes[0].legend(fontsize=8)
for ax in axes:
    ax.set_xlabel("update step")
axes[0].set_ylabel("coefficient of fixed observation")
plt.show()

# %% [markdown]
# ## 4. Colored noise and the Toeplitz whitener
#
# Everything so far relied on one hidden assumption: *variance ranking finds
# signal because the noise contributes equally to every direction*. Colored
# (correlated) noise breaks this — noisy directions can out-rank informative
# ones, and the shrinkage threshold sits at the wrong level.
#
# We keep the **same bump model** as Section 2 and change only the noise:
# stationary noise drawn from a non-trivial power spectral density (a red
# $1/f$ component, a spectral bump, and a white floor). Stationary noise on a
# uniform grid has Toeplitz covariance and is diagonal in Fourier/Hartley
# space, which is exactly the structure `ToeplitzWhitener` estimates: it
# learns the per-frequency noise variance from noise realizations
# ($x - \text{signal}$, provided during training) and divides it out.

# %%
torch.manual_seed(7)

freq = torch.minimum(torch.arange(N), N - torch.arange(N)).double()
S_psd = 0.2 + 5.0 / (1.0 + freq) ** 1.5 + 4.0 * torch.exp(-0.5 * ((freq - 12.0) / 2.0) ** 2)

H = hartley_transform(torch.eye(N))          # orthogonal Hartley matrix
C_col = H @ torch.diag(S_psd) @ H            # Toeplitz noise covariance


def simulate_colored(m, s=0.4):
    theta = theta0_bump + (2 * torch.rand(m, P) - 1) * (s * delta_base)
    signal = bump_batch(theta)
    noise = hartley_transform(torch.randn(m, N) * torch.sqrt(S_psd))
    return signal + noise, signal


fig, axes = plt.subplots(1, 2, figsize=(9, 2.8), constrained_layout=True)
axes[0].semilogy(freq[: N // 2], S_psd[: N // 2], color="C3")
axes[0].axhline(1.0, color="0.5", ls="--", lw=0.8)
axes[0].set_xlabel("frequency index")
axes[0].set_ylabel("noise PSD")
axes[0].set_title("noise power spectrum")
x_col, f_col = simulate_colored(1)
axes[1].plot(t, x_col[0], color="0.6", lw=0.8, label="data")
axes[1].plot(t, f_col[0], color="k", label="signal")
axes[1].legend(fontsize=8)
axes[1].set_xlabel("$t$")
axes[1].set_title("one realization (colored noise)")
plt.show()

# %%
K4 = 16

emb_colored = train_dynamic_svd(
    DynamicSVD(n_components=K4, buffer_size=64), simulate_colored
)
emb_whitened = train_dynamic_svd(
    DynamicSVD(n_components=K4, buffer_size=64, whitener=ToeplitzWhitener()),
    simulate_colored,
)

F_true_col = fisher(bump_model, theta0_bump, C_col)
F_emb_colored = fisher_embedded(emb_colored, bump_model, theta0_bump, C_col)
F_emb_whitened = fisher_embedded(emb_whitened, bump_model, theta0_bump, C_col)

print(f"white noise, k = {K4} (Section 2):   r = {ratio_sweep[(0.4, K4)]:.4f}")
print(f"colored noise, no whitener:      r = {info_ratio(F_emb_colored, F_true_col):.4f}")
print(f"colored noise, ToeplitzWhitener: r = {info_ratio(F_emb_whitened, F_true_col):.4f}")

plot_fisher_corner(
    [F_true_col, F_emb_colored, F_emb_whitened],
    ["full data (analytic)", "no whitener", "ToeplitzWhitener"],
    ["k", "C3", "C2"],
    theta0_bump,
    names_bump,
    title="Colored noise: whitening restores the lost information",
)
plt.show()

# %% [markdown]
# Without whitening, the SVD spends components on high-variance noise
# directions and the ellipses inflate visibly. With the Toeplitz whitener the
# embedding operates in units where the noise is white again — the ellipses
# return to the full-data result.
#
# The eigenvalue spectra tell the same story from Section 2's perspective:
# whitening restores the flat unit noise floor, which also puts the Wiener
# shrinkage threshold back at the right level:

# %%
fig, ax = plt.subplots(figsize=(6.5, 3.4), constrained_layout=True)
ax.semilogy(np.arange(1, K4 + 1), emb_colored.eigenvalues, "o-", ms=3,
            color="C3", label="no whitener (raw units)")
ax.semilogy(np.arange(1, K4 + 1), emb_whitened.eigenvalues, "o-", ms=3,
            color="C2", label="ToeplitzWhitener (whitened units)")
ax.axhline(1.0, color="0.5", ls="--", lw=0.8)
ax.text(K4 - 4, 1.2, "noise floor", color="0.4", fontsize=8)
ax.set_xlabel("component index")
ax.set_ylabel(r"eigenvalue $\lambda$")
ax.legend(fontsize=8)
ax.set_title("Whitening restores a flat unit noise floor")
plt.show()

# %% [markdown]
# ## Summary
#
# - **Information lives in the retained subspace.** For a linear compressor,
#   Fisher information is the complete diagnostic: rotations (Procrustes) and
#   per-component rescalings (shrinkage, normalization) never change it.
#   Posterior shape — including non-Gaussian structure — survives compression
#   whenever the informative subspace does.
# - **Choose $k$ generously.** The information-ratio-vs-$k$ curve gives the
#   minimum for a given proposal width; Wiener shrinkage damps surplus
#   noise-floor components automatically, so over-provisioning is cheap.
#   Wider proposals (early adaptive rounds) need more components than narrow
#   ones.
# - **Procrustes stabilization is about trainability, not information.** It
#   keeps the meaning of each embedding output stable while the basis evolves
#   under a narrowing proposal — the input stationarity the downstream
#   network needs.
# - **Whitening is about noise geometry.** Colored noise breaks variance
#   ranking *and* the shrinkage threshold; `ToeplitzWhitener` (stationary
#   noise) or `DiagonalWhitener` (per-bin noise) restore both. Without a
#   whitener, make sure your noise has unit variance — the hard-coded
#   shrinkage floor is tracked in
#   [cweniger/falcon#113](https://github.com/cweniger/falcon/issues/113).
#
# In a falcon graph, `DynamicSVD` is used as (part of) an estimator's
# `embedding:` configuration — see `falcon.embeddings.instantiate_embedding`
# and the `examples/` directory for end-to-end usage.
