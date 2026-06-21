import torch
import torch.nn as nn
from torch.nn.parameter import UninitializedParameter

from falcon.core.logger import log
from falcon.embeddings.builder import Apply


class _RunningNorm(nn.Module):
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
            self.running_mean = x.mean(dim=dim, keepdim=True).detach()
            self.running_var = ((x - self.running_mean) ** 2).mean(
                dim=dim, keepdim=True
            ).detach() + self.epsilon**2
            self.min_variance = self.running_var.clone()
            self.initialized = True

        if self.training:
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

            if self.monotonic_variance:
                self.min_variance = torch.minimum(self.min_variance, self.running_var)

            if self.log_prefix:
                log({
                    "mean_min": self.running_mean.min().item(),
                    "mean_max": self.running_mean.max().item(),
                    "std_min": self.running_var.sqrt().min().item(),
                    "std_max": self.running_var.sqrt().max().item(),
                }, prefix=self.log_prefix)

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


def RunningNorm(inputs=None, *, momentum=0.01, epsilon=1e-20, output_dtype=None,
                dim=(0,), log_prefix=None, adaptive_momentum=False,
                monotonic_variance=True, use_log_update=False):
    """Typed factory for RunningNorm embedding. Returns a pipeline config dict.

    In a notebook, use: RunningNorm(["x"], momentum=0.01)
    In YAML, use:  _target_: falcon.embeddings.RunningNorm  (inputs via _input_)
    """
    return Apply(_RunningNorm, inputs, momentum=momentum, epsilon=epsilon,
                 output_dtype=output_dtype, dim=dim, log_prefix=log_prefix,
                 adaptive_momentum=adaptive_momentum, monotonic_variance=monotonic_variance,
                 use_log_update=use_log_update)


# Backwards compatibility alias
LazyOnlineNorm = RunningNorm


def hartley_transform(x):
    """Hartley transform: H(x) = Re(FFT(x)) - Im(FFT(x)). Its own inverse."""
    fft = torch.fft.fft(x, dim=-1, norm="ortho")
    return fft.real - fft.imag


class _DiagonalWhitener(torch.nn.Module):
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
        """Update running mean and variance from current batch."""
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
        """Apply whitening: (x - mean) / std."""
        if self.use_fourier:
            x = hartley_transform(x)

        std = torch.sqrt(self.running_var + self.eps)
        x_white = (x - self.running_mean) / std

        if self.use_fourier:
            x_white = hartley_transform(x_white)

        return x_white

    def _mean_var_to_scaling(self, mean, var):
        return torch.cat([0.5 * torch.log(var) + 1, mean], dim=-1)

    # TODO: Currently not used anywhere, add tests?
    def get_scaling(self):
        return self._mean_var_to_scaling(self.running_mean, self.running_var)

    # TODO: Currently not used anywhere, add tests?
    def get_logdet_jac(self):
        return torch.log(self.running_var.sqrt()).sum(dim=-1)

    # TODO: Currently not used anywhere, add tests?
    def batch_forward(self, x):
        batch_dim = len(x)

        batch_mean = x.mean(dim=0).detach()
        batch_var = x.var(dim=0, unbiased=False).detach() + self.eps**2

        mean = batch_mean.unsqueeze(0).repeat(batch_dim, 1)
        var = batch_var.unsqueeze(0).repeat(batch_dim, 1)

        mean = mean + torch.randn_like(x) * var**0.5
        var = var * torch.exp(torch.randn_like(x) * 0.5 - 0.5)

        x_scaled = (x - mean) / (var.sqrt() + self.eps)

        log({
            "batch_mean": batch_mean.mean().item(),
            "batch_var": batch_var.mean().item(),
        }, prefix="whitener")

        scaling = self._mean_var_to_scaling(mean, var)
        logdet_jac = torch.log(var.sqrt()).sum(dim=-1)

        return x_scaled, scaling, logdet_jac


def DiagonalWhitener(inputs=None, *, dim, momentum=0.1, eps=1e-8,
                     use_fourier=False, track_mean=True):
    """Typed factory for DiagonalWhitener embedding. Returns a pipeline config dict.

    In a notebook, use: DiagonalWhitener(["x"], dim=100)
    In YAML, use:  _target_: falcon.embeddings.DiagonalWhitener  (inputs via _input_)
    """
    return Apply(_DiagonalWhitener, inputs, dim=dim, momentum=momentum,
                 eps=eps, use_fourier=use_fourier, track_mean=track_mean)
