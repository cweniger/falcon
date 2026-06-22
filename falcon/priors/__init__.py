"""Prior distributions for simulation-based inference.

Provides product priors with bidirectional transformations between
parameter space and latent space.
"""

from falcon.priors.product import Product, TransformedPrior

__all__ = [
    "Product",
    "TransformedPrior",
]
