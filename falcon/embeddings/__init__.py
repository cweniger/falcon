"""Embedding infrastructure for neural network pipelines.

Provides declarative configuration-driven embedding construction,
normalization utilities, and dimensionality reduction.
"""

from falcon.embeddings.builder import instantiate_embedding, EmbeddingWrapper
from falcon.embeddings.norms import LazyOnlineNorm, DiagonalWhitener, hartley_transform
from falcon.embeddings.svd import PCAProjector

__all__ = [
    "instantiate_embedding",
    "EmbeddingWrapper",
    "LazyOnlineNorm",
    "DiagonalWhitener",
    "hartley_transform",
    "PCAProjector",
]
