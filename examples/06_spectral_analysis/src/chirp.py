"""Multi-harmonic chirp signal generator.

Auto-differentiable w.r.t. all continuous parameters via JAX.

Parameters
----------
f0 : initial frequency (Hz)
chirp_mass : controls how fast the frequency sweeps up (arbitrary units)
t_c : coalescence time (s); must be > T_obs
A0 : overall amplitude scale
harmonic_decay : exponential decay rate for higher harmonics;
    amplitude of k-th harmonic ~ exp(-harmonic_decay * (k - 1))

The waveform is built as:
    h(t) = sum_{k=1}^{n_harmonics} A_k(t) * cos(k * phi(t))

where phi(t) = 2*pi * cumulative_trapezoid(f(t)) and

    f(t) = f0 * (1 - t / t_c)^(-3/8 * chirp_mass)   (PN-inspired chirp)
    A(t) = A0 * (1 - t / t_c)^(-1/4)                  (amplitude growth)
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np


def chirp_signal(
    f0: float,
    chirp_mass: float,
    t_c: float,
    A0: float,
    harmonic_decay: float,
    n_harmonics: int = 4,
    N: int = 1_000_000,
    T_obs: float | None = None,
) -> np.ndarray:
    """Generate a multi-harmonic chirp signal.

    Parameters
    ----------
    f0 : float
        Initial frequency (Hz).
    chirp_mass : float
        Chirp mass parameter controlling frequency evolution rate.
        Higher values produce faster chirps.  Units are arbitrary;
        it enters as f(t) = f0 * (1 - t / t_c)^(-3/8 * chirp_mass).
    t_c : float
        Time of coalescence (s).  Must exceed T_obs.
    A0 : float
        Overall amplitude scale.
    harmonic_decay : float
        Exponential decay factor for harmonic amplitudes.
        Harmonic k has amplitude A(t) * exp(-harmonic_decay * (k-1)).
    n_harmonics : int
        Number of harmonics to include (1 = fundamental only).
    N : int
        Number of time-domain samples.
    T_obs : float or None
        Observation duration (s).  Defaults to 0.9 * t_c.

    Returns
    -------
    h : numpy.ndarray, shape (N,)
        Time-domain signal.
    """
    if T_obs is None:
        T_obs = 0.9 * t_c
    h = _chirp_impl(f0, chirp_mass, t_c, A0, harmonic_decay, n_harmonics, N, T_obs)
    return np.asarray(h)


@functools.partial(jax.jit, static_argnums=(5, 6))
def _chirp_impl(f0, chirp_mass, t_c, A0, harmonic_decay, n_harmonics, N, T_obs):
    """JIT-compiled core.  n_harmonics and N are static (integers)."""
    t = jnp.linspace(0.0, T_obs, N)
    dt = T_obs / (N - 1)

    # Dimensionless time to coalescence: tau in (0, 1]
    tau = 1.0 - t / t_c

    # --- Frequency evolution (PN-inspired) ---
    freq_exponent = -3.0 / 8.0 * chirp_mass
    f_t = f0 * tau ** freq_exponent

    # --- Amplitude evolution ---
    A_t = A0 * tau ** (-0.25)

    # --- Phase via trapezoidal cumulative integration ---
    trapezoids = (f_t[:-1] + f_t[1:]) * 0.5 * dt
    phase = jnp.concatenate([jnp.zeros(1), jnp.cumsum(trapezoids)])
    phase = 2.0 * jnp.pi * phase

    # --- Sum over harmonics ---
    def _add_harmonic(h, k):
        amp_k = A_t * jnp.exp(-harmonic_decay * (k - 1))
        h = h + amp_k * jnp.cos(k * phase)
        return h, None

    ks = jnp.arange(1, n_harmonics + 1)
    h, _ = jax.lax.scan(_add_harmonic, jnp.zeros(N), ks)

    return h
