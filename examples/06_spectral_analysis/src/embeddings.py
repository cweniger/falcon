"""Embedding modules for the spectral analysis example.

- SVDEmbedding: Scaffolded streaming SVD embedding using fuge.StreamingPCA
- Tokenizer: Simulator wrapper around fuge.ChirpTokenizer for use as a falcon node
- TokenEmbed: Adapter for fuge.ChirpTokenEmbedding that returns only the tensor
- SpectralTokenEmbed: Spectral (Fourier feature) embedding for token frequencies
- Concat: Concatenation along last dimension for combining embedding streams
"""

import math
import torch
import torch.nn as nn
from fuge.spectral import ChirpTokenizer, ChirpTokenEmbedding


class SVDEmbedding(nn.Module):
    """Scaffolded SVD embedding for raw signals.

    During training, uses clean signal (y) and noise (n) scaffolds to
    learn whitening and PCA. At inference, applies learned projection
    to the noisy observation (x) only.
    """

    def __init__(self, N=100_000, n_components=32, buffer_size=256, momentum=0.1):
        super().__init__()
        from falcon.embeddings import DiagonalWhitener
        from fuge.svd import StreamingPCA
        self.N = N
        self.n_components = n_components
        self.whitener = DiagonalWhitener(N, momentum=momentum, track_mean=False)
        self.projector = StreamingPCA(n_components=n_components, buffer_size=buffer_size,
                                      momentum=momentum, shrinkage=True)
        self.mlp = nn.Sequential(
            nn.LazyLinear(128), nn.ReLU(),
            nn.LazyLinear(64),
        )
        self.double()

    def forward(self, x, y=None, n=None):
        x = x.double()
        if y is not None and self.training:
            y, n = y.double(), n.double()
            self.whitener.update(y + n)
            white_y = self.whitener(y)
            self.projector.update(white_y)
        x1 = self.whitener(x)
        coeffs = self.projector(x1)  # (batch, n_components)
        return self.mlp(coeffs)


class Tokenizer:
    """Simulator wrapper around fuge.ChirpTokenizer for use as a falcon node.

    Takes raw signals (numpy) and returns spectral tokens (numpy).
    Used as a deterministic intermediate node between x and theta.
    Output shape: (B, N, 9) where N = n_windows * n_peaks.
    Fields: score, t_start, t_end, f_start, f_end, A_start, A_end, phase_start, phase_end.
    """

    def __init__(self, k=1024, n_peaks=3, n_dlnf=11, dlnf_min=0.0, dlnf_max=0.05):
        self.tokenizer = ChirpTokenizer(k=k, n_peaks=n_peaks, n_dlnf=n_dlnf,
                                        dlnf_min=dlnf_min, dlnf_max=dlnf_max
                                        ).cpu().double()

    def simulate_batch(self, batch_size, x):
        x_tensor = torch.tensor(x, dtype=torch.float64, device="cpu")
        with torch.no_grad():
            tokens = self.tokenizer(x_tensor)
        return tokens.data.numpy()


class TokenEmbed(nn.Module):
    """Adapter for fuge.ChirpTokenEmbedding.

    Wraps ChirpTokenEmbedding with sensible defaults for the spectral
    analysis example. Normalization is handled by an explicit RunningNorm
    layer in the embedding pipeline (configured in YAML).

    Args:
        time_range: (t_min, t_max) sample-index bounds for time embedding.
        freq_resolution: frequency resolution in cycles/sample.
        amp_range: (A_min, A_max) for amplitude embedding.
        phase_range: max unwrapped phase extent (radians).
    """

    def __init__(self, time_range=(0, 100000), freq_resolution=1e-4,
                 amp_range=(0.0, 10.0), phase_range=1000.0):
        super().__init__()
        from fuge.spectral.embedding import (
            HarmonicEmbeddingConfig, HarmonicPhaseEmbeddingConfig,
        )
        t_min, t_max = time_range
        self.embed = ChirpTokenEmbedding(
            time=HarmonicEmbeddingConfig(t_min, t_max, resolution=(t_max - t_min) / 100),
            freq=HarmonicEmbeddingConfig(0.0, 0.5, resolution=freq_resolution),
            amp=HarmonicEmbeddingConfig(amp_range[0], amp_range[1],
                                        resolution=(amp_range[1] - amp_range[0]) / 20),
            phase=HarmonicPhaseEmbeddingConfig(phi_max=phase_range, phi_resolution=0.01),
        ).double()

    def forward(self, raw_tokens):
        """Embed chirp tokens.

        Args:
            raw_tokens: Tensor of shape (B, N, 9) from ChirpTokenizer.

        Returns:
            Tensor of shape (B, N, n_embed), float64.
        """
        return self.embed(raw_tokens.detach().double())


class SpectralTokenEmbed(nn.Module):
    """Spectral (Fourier feature) embedding for chirp tokens.

    Converts raw ChirpTokenizer output (B, N, 9) into per-token feature
    vectors suitable for a transformer. Frequencies are mapped to
    Fourier features; phases are encoded as cos/sin pairs; SNR and time
    are appended as scalars.

    Input field layout (ChirpTokenizer, 9 fields):
      score, t_start, t_end, f_start, f_end, A_start, A_end, phase_start, phase_end

    Feature layout per token (output):
      - f_start raw (normalized to [-1,1])         -> 1
      - f_end   raw (normalized to [-1,1])         -> 1
      - f_start spectral: sin/cos at n_freq scales -> 2 * n_freq
      - f_end   spectral: sin/cos at n_freq scales -> 2 * n_freq
      - log-SNR (log1p(score) / amp_scale)         -> 1
      - time position (t_start / N_signal * 2 - 1) -> 1
      - phase_start: cos(ps), sin(ps)              -> 2
      - phase_end:   cos(pe), sin(pe)              -> 2  (boundary mode)
      Total n_embed = 4 * n_freq + 8 (boundary) or 4 * n_freq + 6 (center)

    Args:
        n_freq: Number of Fourier scales for frequency embedding. Default 8.
        phase_mode: "boundary" keeps both phase endpoints (default),
            "center" averages them.
        amp_scale: Divisor for log1p(score) normalization.
        N_signal: Total signal length in samples, used to normalize t_start.
    """

    def __init__(self, n_freq=8, phase_mode="boundary", amp_scale=10.0, N_signal=100000):
        super().__init__()
        self.n_freq = n_freq
        self.phase_mode = phase_mode
        self.amp_scale = amp_scale
        self.N_signal = N_signal
        # Geometrically spaced scales: pi, 2pi, 4pi, ..., pi*2^(n_freq-1)
        scales = math.pi * (2.0 ** torch.arange(n_freq, dtype=torch.float64))
        self.register_buffer("scales", scales)
        # raw f_start + f_end (2) + spectral (4*n_freq) + snr (1) + time (1) + phases
        self.n_embed = 4 * n_freq + (8 if phase_mode == "boundary" else 6)

    def _freq_embed(self, f):
        """Fourier features for a frequency value in [-1, 1].

        Args:
            f: (...) shaped tensor.
        Returns:
            (..., 2*n_freq) shaped tensor of [sin, cos] pairs.
        """
        scaled = f.unsqueeze(-1) * self.scales
        return torch.cat([torch.sin(scaled), torch.cos(scaled)], dim=-1)

    def forward(self, raw_tokens):
        """Embed chirp tokens with spectral frequency features.

        Args:
            raw_tokens: (B, N, 9) from ChirpTokenizer (already flat).
                Fields: score, t_start, t_end, f_start, f_end,
                        A_start, A_end, phase_start, phase_end.

        Returns:
            (B, N, n_embed) float64 tensor.
        """
        raw_tokens = raw_tokens.detach().double()
        B, N, _ = raw_tokens.shape

        # f_start/f_end in cycles/sample [0, 0.5] -> normalize to [-1, 1]
        f_start = raw_tokens[..., 3] * 4 - 1
        f_end = raw_tokens[..., 4] * 4 - 1
        snr = torch.log1p(raw_tokens[..., 0]) / self.amp_scale
        # t_start in sample indices -> normalize to [-1, 1]
        t = raw_tokens[..., 1] / self.N_signal * 2 - 1
        ps = raw_tokens[..., 7]
        pe = raw_tokens[..., 8]

        parts = [
            f_start.unsqueeze(-1),        # (B, N, 1) normalized frequency
            f_end.unsqueeze(-1),          # (B, N, 1) normalized frequency
            self._freq_embed(f_start),    # (B, N, 2*n_freq)
            self._freq_embed(f_end),      # (B, N, 2*n_freq)
            snr.unsqueeze(-1),            # (B, N, 1) log-SNR
            t.unsqueeze(-1),              # (B, N, 1) time position
        ]

        if self.phase_mode == "boundary":
            parts += [
                torch.stack([torch.cos(ps), torch.sin(ps)], dim=-1),
                torch.stack([torch.cos(pe), torch.sin(pe)], dim=-1),
            ]
        else:
            phi = (ps + pe) / 2
            parts.append(torch.stack([torch.cos(phi), torch.sin(phi)], dim=-1))

        return torch.cat(parts, dim=-1)  # (B, N, n_embed)


class Concat(nn.Module):
    """Concatenate multiple inputs along the last dimension."""

    def forward(self, *inputs):
        return torch.cat(inputs, dim=-1)
