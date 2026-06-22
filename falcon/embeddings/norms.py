import torch
import torch.nn as nn
from torch.nn.parameter import UninitializedParameter

from falcon.core.logger import log


class RunningNorm(nn.Module):
    def __init__(
        self,
        momentum=0.01,
        epsilon=1e-20,
        output_dtype=None,
        dim=(0,),
        log_prefix=None,
        adaptive_momentum=False,
        monotonic_variance=True,
        use_log_update=False,
    ):
        super().__init__()
        self.momentum = momentum
        self.epsilon = epsilon
        self.output_dtype = getattr(torch, output_dtype) if isinstance(output_dtype, str) else output_dtype
        self.dim = tuple(dim) if isinstance(dim, (list, tuple)) else (dim,)
        self.log_prefix = log_prefix
        self.monotonic_variance = monotonic_variance
        self.use_log_update = use_log_update
        self.adaptive_momentum = adaptive_momentum

        self.register_buffer("running_mean", None)
        self.register_buffer("running_var", None)
        self.register_buffer("min_variance", None)

        self.initialized = False

    def forward(self, x):
        dim = self.dim

        if not self.initialized:
            # Initialize running statistics based on the first minibatch
            self.running_mean = x.mean(dim=dim, keepdim=True).detach()
            self.running_var = ((x - self.running_mean) ** 2).mean(
                dim=dim, keepdim=True
            ).detach() + self.epsilon**2
            self.min_variance = self.running_var.clone()
            self.initialized = True

        if self.training:
            # Compute batch mean and variance over specified dims
            batch_mean = x.mean(dim=dim, keepdim=True).detach()
            batch_var = ((x - self.running_mean) ** 2).mean(dim=dim, keepdim=True).detach()

            if self.adaptive_momentum:
                threshold_ratio = 2
                beta = 0.1
                momentum_eff = self.momentum * torch.sigmoid(
                    ((self.running_var / batch_var) ** 0.5 - threshold_ratio) / beta
                )
            else:
                momentum_eff = self.momentum

            # Update running statistics (match shape explicitly)
            self.running_mean = (
                1 - momentum_eff
            ) * self.running_mean + momentum_eff * batch_mean
            if self.use_log_update:
                self.running_var = torch.exp(
                    (1 - momentum_eff) * torch.log(self.running_var)
                    + momentum_eff * torch.log(batch_var)
                )
            else:
                self.running_var = (
                    1 - momentum_eff
                ) * self.running_var + momentum_eff * batch_var

            # Update minimum variance if monotonic_variance is enabled
            if self.monotonic_variance:
                self.min_variance = torch.minimum(self.min_variance, self.running_var)

            # Log normalization statistics
            if self.log_prefix:
                log({
                    "mean_min": self.running_mean.min().item(),
                    "mean_max": self.running_mean.max().item(),
                    "std_min": self.running_var.sqrt().min().item(),
                    "std_max": self.running_var.sqrt().max().item(),
                }, prefix=self.log_prefix)
        # Use minimum variance for normalization if monotonic_variance is enabled
        effective_var = (
            self.min_variance if self.monotonic_variance else self.running_var
        )
        result = (x - self.running_mean) / (effective_var.sqrt() + self.epsilon)
        if self.output_dtype is not None:
            result = result.to(self.output_dtype)
        return result

    def inverse(self, x):
        effective_var = (
            self.min_variance if self.monotonic_variance else self.running_var
        )
        return x * effective_var.sqrt() + self.running_mean

    def volume(self):
        effective_var = (
            self.min_variance if self.monotonic_variance else self.running_var
        )
        return effective_var.sqrt().prod()


# Backwards compatibility alias
LazyOnlineNorm = RunningNorm


def hartley_transform(x):
    """
    Hartley transform: H(x) = Re(FFT(x)) - Im(FFT(x))
    It is its own inverse: H(H(x)) = x
    """
    fft = torch.fft.fft(x, dim=-1, norm="ortho")
    return fft.real - fft.imag


class ToeplitzWhitener(torch.nn.Module):
    """Whitener for 1D time series assuming stationary (Toeplitz) noise covariance.

    Estimates per-frequency variance via EMA in Hartley space and whitens by
    dividing by the estimated std. Noise is assumed zero-mean.

    update(noise)  — update EMA variance from a batch of noise samples
    __call__(x)    — whiten x (Hartley → divide by std → inverse Hartley)
    """

    def __init__(self, momentum: float = 0.1, eps: float = 1e-8) -> None:
        super().__init__()
        self.momentum = momentum
        self.eps = eps
        self.register_buffer("running_var", None)
        self.initialized = False

    def update(self, noise: torch.Tensor) -> None:
        """Update EMA variance from noise samples of shape (batch_size, T)."""
        h = hartley_transform(noise)
        batch_var = h.var(dim=0, unbiased=False).detach()
        if not self.initialized:
            self.running_var = batch_var
            self.initialized = True
        else:
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Whiten x of shape (batch_size, T)."""
        h = hartley_transform(x)
        h_white = h / torch.sqrt(self.running_var + self.eps)
        return hartley_transform(h_white)


class DiagonalWhitener(torch.nn.Module):
    def __init__(self, dim, momentum=0.1, eps=1e-8, use_fourier=False, track_mean=True):
        """
        dim: number of features (last dimension of x)
        momentum: how much of the new batch stats to use (PyTorch-style)
        eps: small constant for numerical stability
        use_fourier: if True, apply Hartley transform before whitening
        """
        super().__init__()
        self.dim = dim
        self.momentum = momentum
        self.eps = eps
        self.use_fourier = use_fourier
        self.track_mean = track_mean

        self.register_buffer("running_mean", torch.zeros(dim))
        self.register_buffer("running_var", torch.ones(dim))
        self.initialized = False

    def update(self, x):
        """
        Update running mean and variance from current batch
        x: Tensor of shape (batch_size, dim)
        """
        if self.use_fourier:
            x = hartley_transform(x)

        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)

        if not self.initialized:
            self.running_mean = batch_mean.detach()
            self.running_var = batch_var.detach()
            self.initialized = True
        else:
            self.running_mean = (
                1 - self.momentum
            ) * self.running_mean + self.momentum * batch_mean.detach()
            self.running_var = (
                1 - self.momentum
            ) * self.running_var + self.momentum * batch_var.detach()
        if not self.track_mean:
            self.running_mean *= 0

    def __call__(self, x):
        """
        Apply whitening: (x - mean) / std
        If use_fourier, whitening happens in Hartley space and is transformed back.
        """
        if self.use_fourier:
            x = hartley_transform(x)

        std = torch.sqrt(self.running_var + self.eps)
        x_white = (x - self.running_mean) / std

        if self.use_fourier:
            x_white = hartley_transform(x_white)

        return x_white

