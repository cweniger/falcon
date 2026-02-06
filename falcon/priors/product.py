"""Priors with latent space transformations.

Provides:
- TransformedPrior: Base class defining forward/inverse interface with mode parameter
- Product: Product of independent marginal distributions
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
      - "hypercube": Maps to/from bounded hypercube. Use with Flow estimator.
      - "standard_normal": Maps to/from N(0, I). Use with Gaussian estimator.

    This base class is used for type checking in estimators like Gaussian
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


class Product(TransformedPrior):
    """
    Maps between target distributions and a latent space (hypercube or standard normal).

    Supports bi-directional transformation with mode selection at call time:
      - forward(z, mode): latent space -> target distribution
      - inverse(x, mode): target distribution -> latent space

    Modes:
      - "hypercube": Maps to/from hypercube domain (default [-2, 2]). Use with Flow estimator.
      - "standard_normal": Maps to/from N(0, I). Use with Gaussian estimator.

    Supported distribution types and their required parameters:
      - "uniform": Linear mapping. Parameters: low, high.
      - "cosine": Uses acos transform for pdf ∝ sin(angle). Parameters: low, high.
      - "sine": Uses asin transform. Parameters: low, high.
      - "uvol": Uniform-in-volume. Parameters: low, high.
      - "normal": Normal distribution. Parameters: mean, std.
      - "triangular": Triangular distribution. Parameters: a (min), c (mode), b (max).
      - "fixed": Fixed value (excluded from latent space). Parameters: value.

    Example:
        prior = Product([
            ("uniform", -100.0, 100.0),
            ("fixed", 5.0),              # Fixed parameter, not in latent space
            ("normal", 0.0, 1.0),
        ])

        # Latent space has dim=2 (only free params)
        # Output space has dim=3 (includes fixed params)

        # For Gaussian estimator (standard normal latent space)
        z = prior.inverse(theta, mode="standard_normal")  # theta: (..., 3) -> z: (..., 2)
        theta = prior.forward(z, mode="standard_normal")  # z: (..., 2) -> theta: (..., 3)

        # For Flow estimator (hypercube latent space)
        u = prior.inverse(theta, mode="hypercube")
        theta = prior.forward(u, mode="hypercube")
    """

    def __init__(self, priors=[], hypercube_range=[-2, 2]):
        """
        Initialize Product.

        Args:
            priors: List of tuples (dist_type, param1, param2, ...).
            hypercube_range: Range for hypercube mode (default: [-2, 2]).
        """
        self.priors = priors
        self.hypercube_range = hypercube_range

        # Separate fixed and free parameters
        self._free_indices = []
        self._fixed_indices = []
        self._fixed_values = {}
        for i, prior in enumerate(priors):
            dist_type = prior[0]
            if dist_type == "fixed":
                self._fixed_indices.append(i)
                self._fixed_values[i] = prior[1]
            else:
                self._free_indices.append(i)

        self._param_dim = len(self._free_indices)  # Latent space dimension
        self._full_param_dim = len(priors)  # Full output dimension

    @property
    def param_dim(self) -> int:
        """Dimension of the latent space (free parameters only)."""
        return self._param_dim

    @property
    def full_param_dim(self) -> int:
        """Dimension of the full parameter space (including fixed parameters)."""
        return self._full_param_dim

    # ==================== Public API ====================

    def forward(self, z, mode="hypercube"):
        """
        Map from latent space to target distribution.

        Args:
            z: Tensor of shape (..., param_dim) in latent space (free params only).
            mode: "hypercube" or "standard_normal".

        Returns:
            Tensor of shape (..., full_param_dim) in target distribution space.
        """
        # Handle case with no free parameters
        if self._param_dim == 0:
            batch_shape = z.shape[:-1] if z.dim() > 1 else (1,)
            result = torch.zeros(*batch_shape, self._full_param_dim, dtype=z.dtype, device=z.device)
            for idx, val in self._fixed_values.items():
                result[..., idx] = val
            return result

        if mode == "standard_normal":
            # Try direct transforms first (avoids CDF precision issues at tails)
            transformed = [None] * self._full_param_dim
            use_direct = True
            z_idx = 0
            for i, prior in enumerate(self.priors):
                dist_type, *params = prior
                if dist_type == "fixed":
                    transformed[i] = torch.full(z.shape[:-1], params[0], dtype=z.dtype, device=z.device)
                else:
                    x_i = self._from_standard_normal(z[..., z_idx], dist_type, *params)
                    if x_i is None:
                        use_direct = False
                        break
                    transformed[i] = x_i
                    z_idx += 1
            if use_direct:
                return torch.stack(transformed, dim=-1)
            # Fall through to CDF approach if any distribution lacks direct transform
            u = self._normal_to_uniform(z)
        elif mode == "hypercube":
            u = self._hypercube_to_uniform(z)
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'hypercube' or 'standard_normal'.")

        # Map [0, 1] to target distributions (CDF approach)
        epsilon = 1e-10  # Supports ~6 sigma tails in float64
        u = torch.clamp(u, epsilon, 1.0 - epsilon)

        transformed = []
        u_idx = 0
        for i, prior in enumerate(self.priors):
            dist_type, *params = prior
            if dist_type == "fixed":
                x_i = torch.full(u.shape[:-1], params[0], dtype=u.dtype, device=u.device)
            else:
                x_i = self._forward_transform(u[..., u_idx], dist_type, *params)
                u_idx += 1
            transformed.append(x_i)

        return torch.stack(transformed, dim=-1)

    def inverse(self, x, mode="hypercube"):
        """
        Map from target distribution to latent space.

        Args:
            x: Tensor of shape (..., full_param_dim) in target distribution space.
            mode: "hypercube" or "standard_normal".

        Returns:
            Tensor of shape (..., param_dim) in latent space (free params only).
        """
        # Handle case with no free parameters
        if self._param_dim == 0:
            batch_shape = x.shape[:-1] if x.dim() > 1 else (1,)
            return torch.zeros(*batch_shape, 0, dtype=x.dtype, device=x.device)

        if mode == "standard_normal":
            # Try direct transforms first (avoids CDF precision issues at tails)
            transformed = []
            use_direct = True
            for i, prior in enumerate(self.priors):
                dist_type, *params = prior
                if dist_type == "fixed":
                    continue  # Skip fixed parameters
                z_i = self._to_standard_normal(x[..., i], dist_type, *params)
                if z_i is None:
                    use_direct = False
                    break
                transformed.append(z_i)
            if use_direct:
                return torch.stack(transformed, dim=-1)
            # Fall through to CDF approach if any distribution lacks direct transform

        # Map target distributions to [0, 1] (CDF approach, free params only)
        uniform = []
        for i, prior in enumerate(self.priors):
            dist_type, *params = prior
            if dist_type == "fixed":
                continue  # Skip fixed parameters
            u_i = self._inverse_transform(x[..., i], dist_type, *params)
            uniform.append(u_i)

        u = torch.stack(uniform, dim=-1)

        # Clamp to avoid numerical issues at boundaries
        epsilon = 1e-10  # Supports ~6 sigma tails in float64
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
            numpy array of shape (batch_size, full_param_dim) in target distribution space.
        """
        # Sample uniform for free parameters only
        u = torch.rand(batch_size, self._param_dim, dtype=torch.float64)

        transformed = []
        u_idx = 0
        for i, prior in enumerate(self.priors):
            dist_type, *params = prior
            if dist_type == "fixed":
                x_i = torch.full((batch_size,), params[0], dtype=torch.float64)
            else:
                x_i = self._forward_transform(u[..., u_idx], dist_type, *params)
                u_idx += 1
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

    # ==================== Direct Standard Normal Transforms ====================
    # These avoid CDF precision issues for unbounded distributions

    @staticmethod
    def _to_standard_normal(x, dist_type, *params):
        """Direct transform to standard normal. Returns None if not supported."""
        if dist_type == "normal":
            mean, std = params
            return (x - mean) / std
        elif dist_type == "lognormal":
            mu, sigma = params
            return (torch.log(x) - mu) / sigma
        else:
            return None  # Fall back to CDF approach

    @staticmethod
    def _from_standard_normal(z, dist_type, *params):
        """Direct transform from standard normal. Returns None if not supported."""
        if dist_type == "normal":
            mean, std = params
            return mean + std * z
        elif dist_type == "lognormal":
            mu, sigma = params
            return torch.exp(mu + sigma * z)
        else:
            return None  # Fall back to CDF approach

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
