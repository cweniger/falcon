import torch
import math


class HypercubeMappingPrior:
    """
    Maps a set of univariate priors between a hypercube domain and their target distributions.

    This class supports a bi-directional transformation:
      - forward: maps from a hypercube domain (default range [-2, 2]) to the target distributions.
      - inverse: maps from the target distribution domain back to the hypercube.

    Supported distribution types and their required parameters:
      - "uniform": Linear mapping from [0, 1] to [low, high].
                   Parameters: low, high.
      - "cosine": Uses an acos transform for distributions with pdf ∝ sin(angle).
                  Parameters: low, high.
      - "sine": Uses an asin transform for a similar angular mapping.
                Parameters: low, high.
      - "uvol": Uniform-in-volume transformation.
                Parameters: low, high.
      - "normal": Maps using the inverse CDF (probit function) for a normal distribution.
                  Parameters: mean, std.
      - "triangular": Maps to a triangular distribution via its inverse CDF.
                      Parameters: a (min), c (mode), b (max).

    Priors should be provided as a list of tuples:
        (dist_type, param1, param2, ...)
    For example, a uniform prior would be ("uniform", low, high) and a triangular prior would be ("triangular", a, c, b).
    """

    def __init__(self, priors=[], hypercube_range=[-2, 2]):
        """
        Initializes the HypercubeMappingPrior object.

        Args:
            priors (list): List of tuples defining each prior. Each tuple starts with a string specifying
                           the distribution type, followed by its parameters.
            hypercube_range (list or tuple): The range of the hypercube domain (default: [-2, 2]).
        """
        self.priors = priors
        self.param_dim = len(priors)
        self.hypercube_range = hypercube_range

    @staticmethod
    def _forward_transform(u, dist_type, *params):
        """
        Maps a value u ∈ [0,1] to a value x in the target distribution's domain.

        Args:
            u (torch.Tensor or scalar): Input value(s) sampled from Uniform(0,1).
            dist_type (str): Type of target distribution.
            *params: Parameters defining the distribution.

        Returns:
            torch.Tensor or scalar: The transformed value x in the target distribution's domain.
        """
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
            # For a normal distribution, params are: mean, std.
            mean, std = params
            # Inverse CDF for standard normal: x = sqrt(2)*erfinv(2*u - 1)
            # Scale and shift: x = mean + std * sqrt(2)*erfinv(2*u - 1)
            return mean + std * math.sqrt(2) * torch.erfinv(2 * u - 1)

        elif dist_type == "triangular":
            # For a triangular distribution, params are: a (min), c (mode), b (max).
            a, c, b = params
            # Calculate threshold = (c - a) / (b - a)
            threshold = (c - a) / (b - a)
            # Piecewise inverse CDF:
            # If u < threshold: x = a + sqrt(u*(b-a)*(c-a))
            # Else: x = b - sqrt((1-u)*(b-a)*(b-c))
            x = torch.where(
                u < threshold,
                a + torch.sqrt(u * (b - a) * (c - a)),
                b - torch.sqrt((1 - u) * (b - a) * (b - c)),
            )
            return x

        else:
            raise ValueError(f"Unknown dist_type: {dist_type}")

    @staticmethod
    def _inverse_transform(x, dist_type, *params):
        """
        Maps a value x in the target distribution's domain back to a value u ∈ [0,1].

        Args:
            x (torch.Tensor or scalar): Input value(s) from the target distribution.
            dist_type (str): Type of target distribution.
            *params: Parameters defining the distribution.

        Returns:
            torch.Tensor or scalar: The corresponding value u ∈ [0,1].
        """
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
            # Compute u from the CDF of the normal distribution:
            # u = (erf((x-mean)/(std*sqrt2)) + 1)/2
            return (torch.erf((x - mean) / (std * math.sqrt(2))) + 1) / 2

        elif dist_type == "triangular":
            a, c, b = params
            # Piecewise CDF:
            # If x < c: u = ((x-a)^2) / ((b-a)*(c-a))
            # Else: u = 1 - ((b-x)^2) / ((b-a)*(b-c))
            u = torch.where(
                x < c,
                ((x - a) ** 2) / ((b - a) * (c - a)),
                1 - ((b - x) ** 2) / ((b - a) * (b - c)),
            )
            return u

        else:
            raise ValueError(f"Unknown dist_type: {dist_type}")

    def forward(self, u):
        """
        Applies the forward transformation to a batch of input values.

        The input tensor u should have shape (..., n_params), where the last dimension
        corresponds to different parameters (each in the hypercube_range). First, the values
        are rescaled to [0,1] and then mapped into the corresponding target distributions.

        Args:
            u (torch.Tensor): Tensor of shape (..., n_params) with values in the hypercube_range.

        Returns:
            torch.Tensor: Tensor of shape (..., n_params) with values in the target distribution domains.
        """
        # Rescale u from hypercube_range to [0, 1]
        u = (u - self.hypercube_range[0]) / (
            self.hypercube_range[1] - self.hypercube_range[0]
        )
        epsilon = 1e-6
        u = torch.clamp(u, epsilon, 1.0 - epsilon).double()

        transformed_list = []
        for i, prior in enumerate(self.priors):
            dist_type = prior[0]
            params = prior[1:]  # Support arbitrary number of parameters per prior
            u_i = u[..., i]
            x_i = self._forward_transform(u_i, dist_type, *params)
            transformed_list.append(x_i)

        return torch.stack(transformed_list, dim=-1)

    def inverse(self, x):
        """
        Applies the inverse transformation to a batch of values from the target distributions.

        The input tensor x should have shape (..., n_params). Each value is mapped back to [0,1]
        and then rescaled to the hypercube_range.

        Args:
            x (torch.Tensor): Tensor of shape (..., n_params) with values in the target distribution domains.

        Returns:
            torch.Tensor: Tensor of shape (..., n_params) with values in the hypercube_range.
        """
        inv_list = []
        for i, prior in enumerate(self.priors):
            dist_type = prior[0]
            params = prior[1:]
            x_i = x[..., i]
            u_i = self._inverse_transform(x_i, dist_type, *params)
            inv_list.append(u_i)

        u = torch.stack(inv_list, dim=-1)
        u = (
            u * (self.hypercube_range[1] - self.hypercube_range[0])
            + self.hypercube_range[0]
        )
        return u

    def simulate_batch(self, batch_size):
        """
        Generates a batch of samples from the target distributions.

        Args:
            batch_size (int): Number of samples to generate.

        Returns:
            torch.Tensor: Tensor of shape (n_samples, n_params) with samples in the target distributions.
        """
        # Generate random samples in the hypercube_range
        u = (
            torch.rand(batch_size, len(self.priors))
            * (self.hypercube_range[1] - self.hypercube_range[0])
            + self.hypercube_range[0]
        )
        u = (
            torch.rand(batch_size, len(self.priors), dtype=torch.float64)
            * (self.hypercube_range[1] - self.hypercube_range[0])
            + self.hypercube_range[0]
        )
        return self.forward(u).numpy()


# ==================== Example Usage ==================== #
if __name__ == "__main__":
    # Define a list of priors.
    # Each prior is a tuple: (distribution type, parameter1, parameter2, ...)
    # Supported examples:
    #  - ("cosine", low, high)
    #  - ("sine", low, high)
    #  - ("uvol", low, high)
    #  - ("uniform", low, high)
    #  - ("normal", mean, std)
    #  - ("triangular", a, c, b)
    priors = [
        ("cosine", 0.0, math.pi),
        ("sine", 0.0, math.pi),
        ("uvol", 100.0, 5000.0),
        ("uniform", 10.0, 10.1),
        ("normal", 0.0, 1.0),
        ("triangular", -1.0, 0.0, 1.0),
    ]

    # Create an instance of HypercubeMappingPrior with the given priors.
    # The hypercube_range is the domain for the input values (default: [-2, 2]).
    hmp = HypercubeMappingPrior(priors)

    # Generate a random tensor 'u' with shape (2, n_params) in the hypercube_range.
    # Here, 2 is the batch size and n_params is the number of priors.
    u = (
        torch.rand(2, len(priors)) * (hmp.hypercube_range[1] - hmp.hypercube_range[0])
        + hmp.hypercube_range[0]
    )

    # Forward transformation: map u from the hypercube domain to the target distribution domains.
    v = hmp.forward(u)

    # Inverse transformation: recover u from the transformed values v.
    w = hmp.inverse(v)

    print("Original u values in hypercube_range:")
    print(u)

    print("\nTransformed v values in target distributions:")
    print(v)

    print("\nRecovered u values from inverse transformation:")
    print(w)

    # Sample generation example
    n_samples = 5
    samples = hmp.sample(n_samples)
    print("\nGenerated samples in the target distributions:")
    print(samples)
