import torch
import torch.nn as nn
from falcon.core.logging import log
from torch.nn.parameter import UninitializedParameter


class LazyOnlineNorm(nn.Module):
    def __init__(
        self,
        momentum=0.01,
        epsilon=1e-20,
        log_prefix=None,
        adaptive_momentum=False,
        monotonic_variance=True,
        use_log_update=False,
    ):
        super().__init__()
        self.momentum = momentum
        self.epsilon = epsilon
        self.log_prefix = log_prefix + ":" if log_prefix else ""
        self.monotonic_variance = monotonic_variance
        self.use_log_update = use_log_update
        self.adaptive_momentum = adaptive_momentum

        self.register_buffer("running_mean", None)
        self.register_buffer("running_var", None)
        self.register_buffer("min_variance", None)

        self.initialized = False

    def forward(self, x):
        if not self.initialized:
            # Initialize running statistics based on the first minibatch
            self.running_mean = x.mean(dim=0).detach()
            self.running_var = ((x - self.running_mean) ** 2).mean(
                dim=0
            ).detach() + self.epsilon**2
            self.min_variance = self.running_var.clone()
            self.initialized = True

        if self.training:
            # Compute batch mean and variance over batch dimension only
            batch_mean = x.mean(dim=0)  # Mean over batch dimension
            batch_var = ((x - self.running_mean) ** 2).mean(dim=0).detach()

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

            log(
                {
                    f"{self.log_prefix}running_mean_{i}": self.running_mean[i].item()
                    for i in range(self.running_mean.shape[0])
                }
            )
            log(
                {
                    f"{self.log_prefix}running_std_{i}": self.running_var[i].item()
                    ** 0.5
                    for i in range(self.running_var.shape[0])
                }
            )
            log(
                {
                    f"{self.log_prefix}batch_mean_{i}": batch_mean[i].item()
                    for i in range(batch_mean.shape[0])
                }
            )
            log(
                {
                    f"{self.log_prefix}batch_std_{i}": batch_var[i].item() ** 0.5
                    for i in range(batch_var.shape[0])
                }
            )

        # Use minimum variance for normalization if monotonic_variance is enabled
        effective_var = (
            self.min_variance if self.monotonic_variance else self.running_var
        )
        return (x - self.running_mean) / (effective_var.sqrt() + self.epsilon)

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


def hartley_transform(x):
    """
    Hartley transform: H(x) = Re(FFT(x)) - Im(FFT(x))
    It is its own inverse: H(H(x)) = x
    """
    fft = torch.fft.fft(x, dim=-1, norm="ortho")
    return fft.real - fft.imag


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

    def _mean_var_to_scaling(self, mean, var):
        return torch.cat([0.5 * torch.log(var) + 1, mean], dim=-1)  # (..., scaling_dim)

    def get_scaling(self):
        return self._mean_var_to_scaling(
            self.running_mean, self.running_var
        )  # (scaling_dim,)
        # return torch.cat([0.1*torch.log(self.running_var)+1, self.running_mean], dim = -1)  # (scaling_dim,)

    def get_logdet_jac(self):
        return torch.log(self.running_var.sqrt()).sum(dim=-1)  # (,)

    def batch_forward(self, x):
        batch_dim = len(x)

        batch_mean = x.mean(dim=0).detach()  # (x_dim,)
        batch_var = x.var(dim=0, unbiased=False).detach() + self.epsilon**2  # (x_dim,)

        mean = batch_mean.unsqueeze(0).repeat(batch_dim, 1)  # (batch_dim, x_dim)
        var = batch_var.unsqueeze(0).repeat(batch_dim, 1)  # (batch_dim, x_dim)

        # Randomize mean and variance per sample
        mean = mean + torch.randn_like(x) * var**0.5
        var = var * torch.exp(torch.randn_like(x) * 0.5 - 0.5)

        x_scaled = (x - mean) / (var.sqrt() + self.epsilon)

        log(
            {
                f"batch_mean {self.tag}": mean[0].mean().item(),
                f"batch_var {self.tag}": var[0].mean().item(),
            }
        )

        scaling = self._mean_var_to_scaling(mean, var)

        logdet_jac = torch.log(var.sqrt()).sum(dim=-1)

        return x_scaled, scaling, logdet_jac
