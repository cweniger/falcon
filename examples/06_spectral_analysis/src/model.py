"""
Simulator components for the spectral analysis example.

  - Signal: noise-free chirp waveform generation via chirp._chirp_impl
  - Noise: independent Gaussian noise generator
  - Data: combiner node (x = y + n)
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np
from chirp import _chirp_impl


class Signal:
    """Noise-free chirp signal generator.

    Generates chirping multi-harmonic signals parameterized by
    (f0, chirp_mass, harmonic_decay). Used as intermediate node: theta → y.
    """

    def __init__(self, N=100_000, t_c=1e6, A0=5.0, n_harmonics=4):
        self.N = N
        self.t_c = t_c
        self.A0 = A0
        self.n_harmonics = n_harmonics

        @functools.partial(jax.jit, static_argnums=(3, 4))
        def _generate_clean(f0, chirp_mass, harmonic_decay,
                            n_harmonics, N, t_c, A0, T_obs):
            return jax.vmap(
                lambda f, m, d: _chirp_impl(f, m, t_c, A0, d, n_harmonics, N, T_obs)
            )(f0, chirp_mass, harmonic_decay)

        self._generate_clean = _generate_clean

    def simulate_batch(self, batch_size, theta):
        T_obs = 0.9 * self.t_c
        signals = self._generate_clean(
            jnp.asarray(theta[:, 0]), jnp.asarray(theta[:, 1]),
            jnp.asarray(theta[:, 2]),
            self.n_harmonics, self.N, self.t_c, self.A0, T_obs,
        )
        return np.asarray(signals, dtype=np.float64)


class Noise:
    """Independent Gaussian noise generator (no parents)."""

    def __init__(self, N=100_000, sigma=1.0):
        self.N = N
        self.sigma = sigma

    def simulate(self):
        return np.float64(np.random.randn(self.N) * self.sigma)


class Data:
    """Combiner node: x = y + n."""

    def simulate(self, y, n):
        return y + n


