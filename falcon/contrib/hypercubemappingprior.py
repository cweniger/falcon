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
    
    def log_prob(self, x):                  ### change 
        """
        计算目标分布参数 x 的对数概率密度。
        
        Args:
            x (torch.Tensor): 形状为 (..., n_params) 的张量，表示目标分布中的参数值。
        
        Returns:
            torch.Tensor: 形状为 (...) 的对数概率密度值。
        """
        log_probs = []
        for i, prior in enumerate(self.priors):
            dist_type = prior[0]
            params = prior[1:]
            x_i = x[..., i]

            if dist_type == "uniform":
                low, high = params
                # 均匀分布: 在 [low, high] 内概率为 1/(high-low)
                log_p = torch.where(
                    (x_i >= low) & (x_i <= high),
                    -torch.log(torch.tensor(high - low, dtype=x_i.dtype, device=x_i.device)),
                    torch.tensor(-torch.inf, dtype=x_i.dtype, device=x_i.device)
                )

            elif dist_type == "cosine":
                low, high = params
                alpha = (x_i - low) / (high - low) * math.pi
                # 概率密度公式: (π / (2*(high-low))) * sin(alpha)
                log_p = torch.log(math.pi / (2 * (high - low))) + torch.log(torch.sin(alpha))
                # 检查有效范围
                valid = (x_i >= low) & (x_i <= high)
                log_p = torch.where(valid, log_p, torch.tensor(-torch.inf, dtype=x_i.dtype, device=x_i.device))

            elif dist_type == "sine":
                low, high = params
                alpha = (x_i - low) / (high - low) * math.pi
                # 概率密度公式: (π / (2*(high-low))) * cos(alpha)
                log_p = torch.log(math.pi / (2 * (high - low))) + torch.log(torch.cos(alpha))
                valid = (x_i >= low) & (x_i <= high)
                log_p = torch.where(valid, log_p, torch.tensor(-torch.inf, dtype=x_i.dtype, device=x_i.device))

            elif dist_type == "uvol":
                low, high = params
                # 概率密度公式: (3x^2) / (high^3 - low^3)
                log_p = torch.log(3 * x_i**2) - torch.log(high**3 - low**3)
                valid = (x_i >= low) & (x_i <= high)
                log_p = torch.where(valid, log_p, torch.tensor(-torch.inf, dtype=x_i.dtype, device=x_i.device))

            elif dist_type == "normal":
                mean, std = params
                # 正态分布对数概率
                log_p = -0.5 * ((x_i - mean) / std)**2 - torch.log(std) - 0.5 * math.log(2 * math.pi)

            elif dist_type == "triangular":
                a, c, b = params
                valid = (x_i >= a) & (x_i <= b)
                # 分段概率密度公式
                term1 = torch.log(2) + torch.log(x_i - a) - torch.log((b - a) * (c - a))
                term2 = torch.log(2) + torch.log(b - x_i) - torch.log((b - a) * (b - c))
                log_p = torch.where(x_i < c, term1, term2)
                log_p = torch.where(valid, log_p, torch.tensor(-torch.inf, dtype=x_i.dtype, device=x_i.device))

            else:
                raise ValueError(f"不支持的分布类型: {dist_type}")

            log_probs.append(log_p)

        # 各维度独立，对数概率相加
        return torch.stack(log_probs, dim=-1).sum(dim=-1)


    @staticmethod
    def _log_abs_det_jac_single_u(u, dist_type, *params, eps=1e-8):
        """
        返回单维在 u∈(0,1) 的 log|dx/du|，不含从 u_raw 到 u 的 1/4 因子。
        注意：这里的 u 是 [0,1] 的，而不是 [-2,2] 的 u_raw。
        """
        if dist_type == "uniform":
            low, high = params
            # x = low + (high-low)*u
            # dx/du = (high-low)
            return torch.log(torch.tensor(high - low, dtype=u.dtype, device=u.device))

        elif dist_type == "cosine":
            low, high = params
            # x = low + (acos(1-2u)/pi)*(high-low)
            # dx/du = (high-low)/(pi*sqrt(u*(1-u)))
            uu = u.clamp(eps, 1.0 - eps)
            return (torch.log(torch.tensor(high - low, dtype=u.dtype, device=u.device))
                    - math.log(math.pi)
                    - 0.5 * (torch.log(uu) + torch.log(1.0 - uu)))

        elif dist_type == "sine":
            low, high = params
            # x = low + (asin(2u-1)+pi/2)*(high-low)/pi
            # dx/du = (high-low)/(pi*sqrt(u*(1-u)))
            uu = u.clamp(eps, 1.0 - eps)
            return (torch.log(torch.tensor(high - low, dtype=u.dtype, device=u.device))
                    - math.log(math.pi)
                    - 0.5 * (torch.log(uu) + torch.log(1.0 - uu)))

        elif dist_type == "uvol":
            low, high = params
            # x = (A u + B)^(1/3), A=high^3-low^3, B=low^3
            # dx/du = A / (3 x^2)
            A = (high**3 - low**3)
            # 用 forward 的 x 以免重复算
            # 但这里没有 x，外层会传进来；所以此函数只负责公式部件，外层提供 x
            raise RuntimeError("uvol 的单维雅可比在外层使用 x 计算，见下方 log_prob_u 实现。")

        elif dist_type == "normal":
            mean, std = params
            # x = mean + std * sqrt(2) * erfinv(2u-1)
            # 设 y = erfinv(2u-1)，dy/du = sqrt(pi)*exp(y^2)
            # dx/du = std*sqrt(2)*sqrt(pi)*exp(y^2) = std*sqrt(2π) * exp(y^2)
            uu = u.clamp(eps, 1.0 - eps)
            y = torch.erfinv(2*uu - 1)
            return (math.log(std) + 0.5*math.log(2*math.pi) + (y**2))

        elif dist_type == "triangular":
            a, c, b = params
            # threshold
            t = (c - a) / (b - a)
            uu = u.clamp(eps, 1.0 - eps)
            # 下支：x = a + sqrt(u*(b-a)*(c-a)) => dx/du = 0.5*sqrt((b-a)*(c-a))/sqrt(u)
            # 上支：x = b - sqrt((1-u)*(b-a)*(b-c)) => dx/du = 0.5*sqrt((b-a)*(b-c))/sqrt(1-u)
            left = 0.5*math.sqrt((b-a)*(c-a)) - 0.5*torch.log(uu)  # 先写错了？我们直接写 log 形式
            # 直接写 log|dx/du|
            log_dx_left  = math.log(0.5) + 0.5*math.log((b-a)*(c-a)) - 0.5*torch.log(uu)
            log_dx_right = math.log(0.5) + 0.5*math.log((b-a)*(b-c)) - 0.5*torch.log(1.0 - uu)
            return torch.where(uu < t, log_dx_left, log_dx_right)

        else:
            raise ValueError(f"Unknown dist_type: {dist_type}")

    def log_prob_u(self, u_raw, eps=1e-8):
        """
        接受 u_raw ∈ [-2,2]（最后一维是参数维度）的张量，返回 u-space 的 log p(u_raw)。
        计算：log p_x(x) + ∑_i [ log|dx_i/du_i| - log 4 ]。
        其中 u = (u_raw+2)/4，x = forward(u_raw) 已含 u_raw→u→x 变换。
        """
        # 1) 先把 u_raw 映到 u ∈ (0,1)
        a, b = self.hypercube_range
        assert a < b
        width = (b - a)  # 这里通常是 4
        u = (u_raw - a) / width
        u = u.clamp(eps, 1.0 - eps)

        # 2) 得到 x（真实参数空间）
        x = self.forward(u_raw)  # 你已有的 forward 会自己做 (u_raw->u->x)

        # 3) 先算 x-space 的 log p(x)
        log_px = self.log_prob(x)  # 你已有的 log_prob(x) 在 x-space

        # 4) 逐维加上 log|dx/du_raw| = log|dx/du| - log(width)
        #    width = 4（若 hypercube_range=[-2,2]）
        per_dim_logs = []
        for i, prior in enumerate(self.priors):
            dist_type = prior[0]
            params = prior[1:]
            u_i = u[..., i]
            x_i = x[..., i]   # 有些分布（uvol）更方便用 x_i

            if dist_type == "uvol":
                # dx/du = A/(3 x^2), 其中 A=high^3-low^3
                low, high = params
                A = (high**3 - low**3)
                # x_i 可能为 0，做个 clamp
                xi2 = (x_i**2).clamp(min=eps)
                log_dx_du = torch.log(torch.tensor(abs(A)/3.0, dtype=u.dtype, device=u.device)) - torch.log(xi2)
            else:
                log_dx_du = self._log_abs_det_jac_single_u(u_i, dist_type, *params, eps=eps)

            # 减去 log(width)
            per_dim_logs.append(log_dx_du - math.log(width))

        log_det = torch.stack(per_dim_logs, dim=-1).sum(dim=-1)

        # 5) 合成 u-space 的 log-prior
        return log_px + log_det


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
