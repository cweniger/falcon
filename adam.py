import math
import torch
from torch.optim import Adam


class TrackingAdam(Adam):
    """
    Adam with additive diffusion for online / drifting regimes.

    This behaves exactly like Adam, but adds a Gaussian noise term
    scaled by the Adam preconditioner to maintain plasticity.

    Intended use:
      - online learning
      - drifting objectives
      - proposal adaptation (e.g. SBI)

    Not intended for:
      - exact posterior sampling
      - static convergence guarantees
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.5, 0.5),
        eps=1e-20,
        weight_decay=0.0,
        diffusion_scale=0.05,
        momentum_gating=True,
        depth_scaling=None,
    ):
        """
        Args:
            diffusion_scale (float):
                Global scale c for diffusion.
                Typical range: 0.01 – 0.1

            momentum_gating (bool):
                Suppress noise when Adam momentum is large.

            depth_scaling (callable or None):
                Optional function f(param) -> scalar multiplier
                to reduce diffusion in early layers.
        """
        super().__init__(
            params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )

        self.diffusion_scale = diffusion_scale
        self.momentum_gating = momentum_gating
        self.depth_scaling = depth_scaling

    @torch.no_grad()
    def step(self, closure=None):
        loss = super().step(closure)

        for group in self.param_groups:
            lr = group["lr"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                if "exp_avg" not in state:
                    continue  # Adam not initialized yet

                m = state["exp_avg"]
                v = state["exp_avg_sq"]

                # Adam preconditioner (with floor for diffusion stability)
                denom = v.sqrt().add_(eps)

                # Base diffusion scale (clamp denom to prevent explosion when v is small)
                denom_for_diffusion = v.sqrt().clamp(min=1e-3).add_(eps)
                sigma = self.diffusion_scale * lr / denom_for_diffusion

                # Optional momentum gating
                if self.momentum_gating:
                    sigma = sigma / (1.0 + m.abs())

                # Optional depth-based scaling
                if self.depth_scaling is not None:
                    sigma = sigma * self.depth_scaling(p)

                # Gaussian diffusion
                sigma = self.diffusion_scale
                noise = torch.randn_like(p)
                p.add_(sigma * noise)

        return loss
