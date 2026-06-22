"""Embedding infrastructure for neural network pipelines.

Provides declarative configuration-driven embedding construction,
normalization utilities, and dimensionality reduction.
"""

from falcon.embeddings.builder import instantiate_embedding, EmbeddingWrapper, apply
from falcon.embeddings.norms import RunningNorm, LazyOnlineNorm, DiagonalWhitener, ToeplitzWhitener, hartley_transform
from falcon.embeddings.svd import PCAProjector, DynamicSVD

__all__ = [
    "apply",
    "instantiate_embedding",
    "EmbeddingWrapper",
    "RunningNorm",
    "LazyOnlineNorm",
    "DiagonalWhitener",
    "ToeplitzWhitener",
    "hartley_transform",
    "PCAProjector",
    "DynamicSVD",
]
