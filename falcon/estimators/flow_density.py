"""Flow networks for density estimation."""

import numpy as np
import torch

import sbi.utils  # Don't remove - needed for sbi.neural_nets.net_builders
from sbi.neural_nets import net_builders

from falcon.embeddings.norms import LazyOnlineNorm


# Network builder registry
NET_BUILDERS = {
    "nsf": net_builders.build_nsf,
    "made": net_builders.build_made,
    "maf": net_builders.build_maf,
    "maf_rqs": net_builders.build_maf_rqs,
    "zuko_nice": net_builders.build_zuko_nice,
    "zuko_maf": net_builders.build_zuko_maf,
    "zuko_nsf": net_builders.build_zuko_nsf,
    "zuko_ncsf": net_builders.build_zuko_ncsf,
    "zuko_sospf": net_builders.build_zuko_sospf,
    "zuko_naf": net_builders.build_zuko_naf,
    "zuko_unaf": net_builders.build_zuko_unaf,
    "zuko_gf": net_builders.build_zuko_gf,
    "zuko_bpf": net_builders.build_zuko_bpf,
}


class FlowDensity(torch.nn.Module):
    """Normalizing flow network with optional parameter normalization."""

    def __init__(
        self,
        theta,
        s,
        theta_norm=False,
        norm_momentum=3e-3,
        net_type="nsf",
        use_log_update=False,
        adaptive_momentum=False,
    ):
        super().__init__()
        self.theta_norm = (
            LazyOnlineNorm(
                momentum=norm_momentum,
                use_log_update=use_log_update,
                adaptive_momentum=adaptive_momentum,
            )
            if theta_norm
            else None
        )

        builder = NET_BUILDERS.get(net_type)
        if builder is None:
            raise ValueError(f"Unknown net_type: {net_type}. Available: {list(NET_BUILDERS.keys())}")
        self.net = builder(theta.float(), s.float(), z_score_x=None, z_score_y=None)

        if self.theta_norm is not None:
            self.theta_norm(theta)  # Initialize normalization stats
        self.scale = 0.2

    def loss(self, theta, s):
        """Compute negative log-likelihood loss."""
        if self.theta_norm is not None:
            theta = self.theta_norm(theta)
        theta = theta.float() * self.scale
        loss = self.net.loss(theta, condition=s.float())
        loss = loss - np.log(self.scale) * theta.shape[-1]
        if self.theta_norm is not None:
            loss = loss + torch.log(self.theta_norm.volume())
        return loss

    def sample(self, num_samples, s):
        """Sample from the flow."""
        samples = self.net.sample((num_samples,), condition=s).detach()
        samples = samples / self.scale
        if self.theta_norm is not None:
            samples = self.theta_norm.inverse(samples).detach()
        return samples

    def log_prob(self, theta, s):
        """Compute log probability."""
        if self.theta_norm is not None:
            theta = self.theta_norm(theta).detach()
        theta = theta * self.scale
        log_prob = self.net.log_prob(theta.float(), condition=s.float())
        log_prob = log_prob + np.log(self.scale) * theta.shape[-1]
        if self.theta_norm is not None:
            log_prob = log_prob - torch.log(self.theta_norm.volume().detach())
        return log_prob
