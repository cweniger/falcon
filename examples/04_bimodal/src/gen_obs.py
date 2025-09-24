# -*- coding: utf-8 -*-

import sys
from pathlib import Path
import numpy as np
import torch
import model

SEED = 0


def main():
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    DIM = getattr(model, "DIM")
    MU = getattr(model, "MU")
    SIGMA = getattr(model, "SIGMA")
    PRIOR_SCALE = getattr(model, "PRIOR_SCALE")
    MEAN_SHIFT = getattr(model, "MEAN_SHIFT")

    if hasattr(model, "PRIOR_LOW") and hasattr(model, "PRIOR_HIGH"):
        prior_low = float(model.PRIOR_LOW)
        prior_high = float(model.PRIOR_HIGH)
    else:
        prior_low = float(MU - PRIOR_SCALE * SIGMA)
        prior_high = float(MU + PRIOR_SCALE * SIGMA)

    # Shrink by 3σ to ensure that z + shift (±3σ) does not exceed the bounds
    low = prior_low + MEAN_SHIFT
    high = prior_high - MEAN_SHIFT

    # 1) samplez_true: torch -> numpy(1,D)
    z_true = torch.distributions.Uniform(low, high).sample((1, DIM)).to(torch.float64)
    z_true_np = z_true.detach().cpu().numpy()

    # 2) m = Signal.simulate_batch(…, z)
    sig = model.Signal()
    m_np = sig.simulate_batch(batch_size=1, z=np.array(z_true_np, copy=True))
    if isinstance(m_np, torch.Tensor):
        m_np = m_np.detach().cpu().numpy()
    if m_np.ndim == 1:
        m_np = m_np[None, :]

    # 3) eps
    noi = model.Noise()
    eps_any = (
        noi.simulate_batch(1) if hasattr(noi, "simulate_batch") else noi.simulate()
    )
    if isinstance(eps_any, torch.Tensor):
        eps_np = eps_any.detach().cpu().numpy()
    else:
        eps_np = np.asarray(eps_any, dtype=np.float64)
    if eps_np.ndim == 1:
        eps_np = eps_np[None, :]

    # 4) x = Add.simulate_batch
    add = model.Add()
    x_np = add.simulate_batch(batch_size=1, m=m_np, n=eps_np)
    if isinstance(x_np, torch.Tensor):
        x_np = x_np.detach().cpu().numpy()
    if x_np.ndim == 1:
        x_np = x_np[None, :]
    x_np = x_np.squeeze(0)  # (D,)
    print("z_true_np", z_true_np.shape)
    print("x_np", x_np.shape)

    # 5) shift 与 bits（for saving）
    shift_np = (m_np - z_true_np).squeeze(0)  # (D,)
    bits_np = (shift_np > 0).astype(np.int8)

    # 6) save（NumPy）
    out = Path(__file__).resolve().parent
    out.mkdir(parents=True, exist_ok=True)

    np.save(out / "../data/x_obs_10dim.npy", x_np)  # (1,D)
    np.save(out / "../data/true_z_10dim.npy", z_true_np.squeeze(0))  # (1,D)
    np.save(out / "../data/shift_10dim.npy", shift_np)  # (D,)
    np.save(out / "../data/eps_10dim.npy", eps_np.squeeze(0))  # (D,)

    # np.savez(out / "obs_pack_10dim.npz",
    #          x_obs=x_np,
    #          z_true=z_true_np.squeeze(0),
    #          shift=shift_np,
    #          eps=eps_np.squeeze(0),
    #          bits=bits_np,
    #          sigma=np.array(SIGMA, dtype=np.float64))


if __name__ == "__main__":
    main()
