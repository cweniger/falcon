"""Prior distributions for simulation-based inference.

Provides hypercube mapping and product priors with bidirectional
transformations between parameter space and latent space.
"""

from falcon.priors.hypercube import Hypercube
from falcon.priors.product import Product, TransformedPrior

# Backward-compat aliases
HypercubeMappingPrior = Hypercube
ProductPrior = Product

__all__ = [
    "Hypercube",
    "Product",
    "TransformedPrior",
    # Backward-compat aliases
    "HypercubeMappingPrior",
    "ProductPrior",
]
