"""
Model components for the spectral analysis example.

Demonstrates falcon + fuge integration for EMRI gravitational wave inference:
  - Simulator: EMRI waveform generation via fuge.emri_signal
  - TokenEmbed: Thin wrapper adapting fuge.ToneTokenEmbedding for falcon's
    sequential embedding pipeline (returns tensor instead of tuple)

The ToneTokenizer and TransformerEmbedding from fuge are referenced directly
in config.yml via _target_ entries — no wrappers needed for those.
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import torch
import torch.nn as nn
from fuge import ToneTokenizer, ToneTokenEmbedding
from fuge.emri import _emri_impl


class Simulator:
    """EMRI gravitational wave simulator with additive Gaussian noise.

    Generates chirping multi-harmonic signals parameterized by
    (f0, chirp_mass, harmonic_decay). Fixed physical parameters
    (coalescence time, amplitude, harmonics) are set at construction.

    Args:
        noise_sigma: Standard deviation of additive Gaussian noise.
        N: Number of time-domain samples per signal.
        t_c: Time of coalescence (seconds).
        A0: Overall amplitude scale.
        n_harmonics: Number of harmonics in the signal.
    """

    def __init__(self, noise_sigma=1.0, N=100_000, t_c=1e6, A0=5.0, n_harmonics=4):
        self.noise_sigma = noise_sigma
        self.N = N
        self.t_c = t_c
        self.A0 = A0
        self.n_harmonics = n_harmonics
        self._rng = jax.random.PRNGKey(0)

        # Vectorized signal generation + noise in one JIT-compiled call
        @functools.partial(jax.jit, static_argnums=(5, 6))
        def _generate_batch(f0, chirp_mass, harmonic_decay, t_c, A0,
                            n_harmonics, N, T_obs, noise_sigma, rng_key):
            signals = jax.vmap(
                lambda f, m, d: _emri_impl(f, m, t_c, A0, d, n_harmonics, N, T_obs)
            )(f0, chirp_mass, harmonic_decay)
            noise = jax.random.normal(rng_key, signals.shape) * noise_sigma
            return signals + noise

        self._generate_batch = _generate_batch

    def simulate_batch(self, batch_size, theta):
        """Generate a batch of noisy EMRI signals.

        Args:
            batch_size: Number of signals to generate.
            theta: Parameter array of shape (batch_size, 3) with columns
                   [f0, chirp_mass, harmonic_decay].

        Returns:
            np.ndarray of shape (batch_size, N) containing noisy signals.
        """
        T_obs = 0.9 * self.t_c
        self._rng, subkey = jax.random.split(self._rng)
        signals = self._generate_batch(
            jnp.asarray(theta[:, 0]), jnp.asarray(theta[:, 1]),
            jnp.asarray(theta[:, 2]), self.t_c, self.A0,
            self.n_harmonics, self.N, T_obs, self.noise_sigma, subkey,
        )
        return np.asarray(signals)


class Tokenizer:
    """Simulator wrapper around fuge.ToneTokenizer for use as a falcon node.

    Takes raw signals (numpy) and returns spectral tokens (numpy).
    Used as a deterministic intermediate node between x and theta.
    """

    def __init__(self, k=1024, n_peaks=3, n_dlnf=11, dlnf_min=0.0, dlnf_max=0.05):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = ToneTokenizer(k=k, n_peaks=n_peaks, n_dlnf=n_dlnf,
                                       dlnf_min=dlnf_min, dlnf_max=dlnf_max
                                       ).to(self.device).double()

    def simulate_batch(self, batch_size, x):
        x_tensor = torch.as_tensor(x, dtype=torch.float64, device=self.device)
        with torch.no_grad():
            tokens = self.tokenizer(x_tensor)
        return tokens.cpu().numpy()


class TokenEmbed(nn.Module):
    """Adapter for fuge.ToneTokenEmbedding that returns only the tensor.

    ToneTokenEmbedding._embed applies cos/sin/log1p feature transforms.
    Normalization is handled by an explicit RunningNorm layer in the
    embedding pipeline (configured in YAML), keeping feature transforms
    separate from normalization — important for sequential SBI where
    the data distribution shifts across rounds.

    Args:
        phase_mode: "center" (n_embed=5) or "boundary" (n_embed=7).
        mask_phases: Zero out phase features (for ablation).
    """

    def __init__(self, phase_mode="center", mask_phases=False):
        super().__init__()
        self.embed = ToneTokenEmbedding(phase_mode=phase_mode, mask_phases=mask_phases).double()

    def forward(self, raw_tokens):
        """Embed raw tokens and return flattened sequence.

        Signal processing (feature extraction) runs in float64.
        Output remains float64; dtype conversion is handled by
        the downstream RunningNorm layer (output_dtype config).

        Args:
            raw_tokens: Tensor of shape (B, W, K, 5) from ToneTokenizer.

        Returns:
            Tensor of shape (B, W*K, n_embed), float64.
        """
        raw_tokens = raw_tokens.detach().double()
        embedded = self.embed._embed(raw_tokens)  # (B, W, K, n_embed)
        B, W, K, n_embed = embedded.shape
        return embedded.reshape(B, W * K, n_embed)
