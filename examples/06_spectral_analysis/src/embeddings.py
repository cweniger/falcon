"""Embedding modules for the spectral analysis example.

- SVDEmbedding: Scaffolded streaming SVD embedding using fuge.StreamingPCA
- Tokenizer: Simulator wrapper around fuge.ToneTokenizer for use as a falcon node
- TokenEmbed: Adapter for fuge.ToneTokenEmbedding that returns only the tensor
"""

import torch
import torch.nn as nn
from fuge.spectral import ToneTokenizer, ToneTokenEmbedding


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
        if self.projector.components.numel() > 0:
            coeffs = self.projector(x1)  # (batch, n_components)
        else:
            coeffs = torch.zeros(x1.shape[0], self.n_components,
                                 dtype=x1.dtype, device=x1.device)
        return self.mlp(coeffs)


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
