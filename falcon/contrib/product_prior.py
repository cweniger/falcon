"""Priors with latent space transformations.

Provides:
- TransformedPrior: Base class defining forward/inverse interface with mode parameter
- ProductPrior: Product of independent marginal distributions
"""

import torch
import math
from abc import ABC, abstractmethod


class TransformedPrior(ABC):
    """Base class for priors that support latent space transformations.

    Subclasses must implement forward() and inverse() with a mode parameter:
      - forward(z, mode): latent space -> parameter space
      - inverse(x, mode): parameter space -> latent space

    Modes:
      - "hypercube": Maps to/from bounded hypercube. Use with SNPE_A.
      - "standard_normal": Maps to/from N(0, I). Use with SNPE_gaussian.

    This base class is used for type checking in estimators like SNPE_gaussian
    that require the transformation interface.
    """

    @property
    @abstractmethod
    def param_dim(self) -> int:
        """Dimension of the parameter space."""
        pass

    @abstractmethod
    def forward(self, z, mode: str = "hypercube"):
        """Transform from latent space to parameter space."""
        pass

    @abstractmethod
    def inverse(self, x, mode: str = "hypercube"):
        """Transform from parameter space to latent space."""
        pass

    @abstractmethod
    def simulate_batch(self, batch_size: int):
        """Sample from the prior distribution."""
        pass


class ProductPrior(TransformedPrior):
    """
    Maps between target distributions and a latent space (hypercube or standard normal).

    Supports bi-directional transformation with mode selection at call time:
      - forward(z, mode): latent space -> target distribution
      - inverse(x, mode): target distribution -> latent space

    Modes:
      - "hypercube": Maps to/from hypercube domain (default [-2, 2]). Use with SNPE_A.
      - "standard_normal": Maps to/from N(0, I). Use with SNPE_gaussian.

    Supported distribution types and their required parameters:
      - "uniform": Linear mapping. Parameters: low, high.
      - "cosine": Uses acos transform for pdf ∝ sin(angle). Parameters: low, high.
      - "sine": Uses asin transform. Parameters: low, high.
      - "uvol": Uniform-in-volume. Parameters: low, high.
      - "normal": Normal distribution. Parameters: mean, std.
      - "triangular": Triangular distribution. Parameters: a (min), c (mode), b (max).

    Example:
        prior = ProductPrior([
            ("uniform", -100.0, 100.0),
            ("normal", 0.0, 1.0),
        ])

        # For SNPE_gaussian (standard normal latent space)
        z = prior.inverse(theta, mode="standard_normal")
        theta = prior.forward(z, mode="standard_normal")

        # For SNPE_A (hypercube latent space)
        u = prior.inverse(theta, mode="hypercube")
        theta = prior.forward(u, mode="hypercube")
    """

    def __init__(self, priors=[], hypercube_range=[-2, 2]):
        """
        Initialize ProductPrior.

        Args:
            priors: List of tuples (dist_type, param1, param2, ...).
            hypercube_range: Range for hypercube mode (default: [-2, 2]).
        """
        self.priors = priors
        self._param_dim = len(priors)
        self.hypercube_range = hypercube_range

    @property
    def param_dim(self) -> int:
        return self._param_dim

    # ==================== Public API ====================

    def forward(self, z, mode="hypercube"):
        """
        Map from latent space to target distribution.

        Args:
            z: Tensor of shape (..., n_params) in latent space.
            mode: "hypercube" or "standard_normal".

        Returns:
            Tensor of shape (..., n_params) in target distribution space.
        """
        # Convert latent space to [0, 1]
        if mode == "hypercube":
            u = self._hypercube_to_uniform(z)
        elif mode == "standard_normal":
            u = self._normal_to_uniform(z)
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'hypercube' or 'standard_normal'.")

        # Map [0, 1] to target distributions
        epsilon = 1e-6
        u = torch.clamp(u, epsilon, 1.0 - epsilon)

        transformed = []
        for i, prior in enumerate(self.priors):
            dist_type, *params = prior
            x_i = self._forward_transform(u[..., i], dist_type, *params)
            transformed.append(x_i)

        return torch.stack(transformed, dim=-1)

    def inverse(self, x, mode="hypercube"):
        """
        Map from target distribution to latent space.

        Args:
            x: Tensor of shape (..., n_params) in target distribution space.
            mode: "hypercube" or "standard_normal".

        Returns:
            Tensor of shape (..., n_params) in latent space.
        """
        # Map target distributions to [0, 1]
        uniform = []
        for i, prior in enumerate(self.priors):
            dist_type, *params = prior
            u_i = self._inverse_transform(x[..., i], dist_type, *params)
            uniform.append(u_i)

        u = torch.stack(uniform, dim=-1)

        # Clamp to avoid numerical issues at boundaries
        epsilon = 1e-6
        u = torch.clamp(u, epsilon, 1.0 - epsilon)

        # Convert [0, 1] to latent space
        if mode == "hypercube":
            return self._uniform_to_hypercube(u)
        elif mode == "standard_normal":
            return self._uniform_to_normal(u)
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'hypercube' or 'standard_normal'.")

    def simulate_batch(self, batch_size):
        """
        Generate samples from the target distributions.

        Args:
            batch_size: Number of samples.

        Returns:
            numpy array of shape (batch_size, n_params) in target distribution space.
        """
        # Sample uniform and transform to target
        u = torch.rand(batch_size, self.param_dim, dtype=torch.float64)

        transformed = []
        for i, prior in enumerate(self.priors):
            dist_type, *params = prior
            x_i = self._forward_transform(u[..., i], dist_type, *params)
            transformed.append(x_i)

        return torch.stack(transformed, dim=-1).numpy()

    # ==================== Latent Space Conversions ====================

    def _hypercube_to_uniform(self, z):
        """Hypercube range -> [0, 1]."""
        low, high = self.hypercube_range
        return (z - low) / (high - low)

    def _uniform_to_hypercube(self, u):
        """[0, 1] -> Hypercube range."""
        low, high = self.hypercube_range
        return u * (high - low) + low

    def _normal_to_uniform(self, z):
        """Standard normal -> [0, 1] via CDF."""
        return 0.5 * (1 + torch.erf(z / math.sqrt(2)))

    def _uniform_to_normal(self, u):
        """[0, 1] -> Standard normal via probit (inverse CDF)."""
        return math.sqrt(2) * torch.erfinv(2 * u - 1)

    # ==================== Distribution Transforms ====================

    @staticmethod
    def _forward_transform(u, dist_type, *params):
        """Map u ∈ [0,1] to target distribution."""
        if dist_type == "uniform":
            low, high = params
            return low + (high - low) * u

        elif dist_type == "cosine":
            low, high = params
            return low + (torch.acos(1 - 2 * u) / math.pi) * (high - low)

        elif dist_type == "sine":
            low, high = params
            return low + (torch.asin(2 * u - 1) + math.pi / 2) * (high - low) / math.pi

        elif dist_type == "uvol":
            low, high = params
            return (((high**3 - low**3) * u) + low**3).pow(1.0 / 3.0)

        elif dist_type == "normal":
            mean, std = params
            return mean + std * math.sqrt(2) * torch.erfinv(2 * u - 1)

        elif dist_type == "triangular":
            a, c, b = params
            threshold = (c - a) / (b - a)
            return torch.where(
                u < threshold,
                a + torch.sqrt(u * (b - a) * (c - a)),
                b - torch.sqrt((1 - u) * (b - a) * (b - c)),
            )

        else:
            raise ValueError(f"Unknown dist_type: {dist_type}")

    @staticmethod
    def _inverse_transform(x, dist_type, *params):
        """Map target distribution to u ∈ [0,1]."""
        if dist_type == "uniform":
            low, high = params
            return (x - low) / (high - low)

        elif dist_type == "cosine":
            low, high = params
            alpha = (x - low) / (high - low) * math.pi
            return (1.0 - torch.cos(alpha)) / 2.0

        elif dist_type == "sine":
            low, high = params
            alpha = (x - low) / (high - low) * math.pi
            return (torch.sin(alpha) + 1.0) / 2.0

        elif dist_type == "uvol":
            low, high = params
            return (x**3 - low**3) / (high**3 - low**3)

        elif dist_type == "normal":
            mean, std = params
            return (torch.erf((x - mean) / (std * math.sqrt(2))) + 1) / 2

        elif dist_type == "triangular":
            a, c, b = params
            return torch.where(
                x < c,
                ((x - a) ** 2) / ((b - a) * (c - a)),
                1 - ((b - x) ** 2) / ((b - a) * (b - c)),
            )

        else:
            raise ValueError(f"Unknown dist_type: {dist_type}")
