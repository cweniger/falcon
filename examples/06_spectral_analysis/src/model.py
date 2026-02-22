"""
Model components for the spectral analysis example.

Demonstrates falcon + fuge integration for EMRI gravitational wave inference:
  - Simulator: EMRI waveform generation via fuge.emri_signal
  - TokenEmbed: Thin wrapper adapting fuge.ToneTokenEmbedding for falcon's
    sequential embedding pipeline (returns tensor instead of tuple)

The ToneTokenizer and TransformerEmbedding from fuge are referenced directly
in config.yml via _target_ entries — no wrappers needed for those.
"""

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import torch
import torch.nn as nn
from fuge import emri_signal, ToneTokenizer, ToneTokenEmbedding


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

    def simulate_batch(self, batch_size, theta):
        """Generate a batch of noisy EMRI signals.

        Args:
            batch_size: Number of signals to generate.
            theta: Parameter array of shape (batch_size, 3) with columns
                   [f0, chirp_mass, harmonic_decay].

        Returns:
            np.ndarray of shape (batch_size, N) containing noisy signals.
        """
        signals = np.empty((batch_size, self.N))
        for i in range(batch_size):
            signals[i] = emri_signal(
                f0=float(theta[i, 0]),
                chirp_mass=float(theta[i, 1]),
                t_c=self.t_c,
                A0=self.A0,
                harmonic_decay=float(theta[i, 2]),
                n_harmonics=self.n_harmonics,
                N=self.N,
            )
        signals += np.random.randn(batch_size, self.N) * self.noise_sigma
        return signals


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

    ToneTokenEmbedding.forward returns (embedded, n_windows, n_peaks),
    but falcon's sequential embedding pipeline expects a single tensor.
    This wrapper discards the auxiliary outputs and lazily computes
    z-score normalization statistics from the first training batch.

    Args:
        phase_mode: "center" (n_embed=5) or "boundary" (n_embed=7).
        mask_phases: Zero out phase features (for ablation).
    """

    def __init__(self, phase_mode="center", mask_phases=False):
        super().__init__()
        self.embed = ToneTokenEmbedding(phase_mode=phase_mode, mask_phases=mask_phases).double()
        self.register_buffer("_norm_initialized", torch.tensor(False))

    def forward(self, raw_tokens):
        """Embed raw tokens and return flattened sequence.

        Signal processing (normalization, feature extraction) runs in float64.
        Output is cast to float32 for the downstream neural network.

        Args:
            raw_tokens: Tensor of shape (B, W, K, 5) from ToneTokenizer.

        Returns:
            Tensor of shape (B, W*K, n_embed), float32.
        """
        raw_tokens = raw_tokens.detach().double()
        if self.training and not self._norm_initialized:
            self.embed.compute_normalization(raw_tokens)
            self._norm_initialized.fill_(True)
        embedded, _, _ = self.embed(raw_tokens)
        return embedded
