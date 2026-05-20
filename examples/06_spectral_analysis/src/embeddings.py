"""Embedding modules for the spectral analysis example.

- SVDEmbedding: Scaffolded streaming SVD embedding using fuge.StreamingPCA
- Tokenizer: Simulator wrapper around fuge.ChirpTokenizer for use as a falcon node
- Concat: Concatenation along last dimension for combining embedding streams
"""

import torch
import torch.nn as nn
from fuge.spectral import ChirpTokenizer


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



class Concat(nn.Module):
    """Concatenate multiple inputs along the last dimension."""

    def forward(self, *inputs):
        return torch.cat(inputs, dim=-1)
