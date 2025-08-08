"""
PCA streaming example model implementation
This file contains model-specific components that can be referenced via _target_
Based on streaming_pca.py example
"""

import torch
import falcon
import wandb

# Global configuration
SIGMA = 1e-3
MU = 0.0
NPAR = 4
NBINS = 100000


class Signal:
    """Signal generator that creates oscillating patterns based on parameters z."""
    
    def __init__(self):
        pass

    def sample(self, num_samples, parent_conditions=[]):
        z = parent_conditions[0]
        z = torch.tensor(z)
        y = torch.linspace(-torch.pi, torch.pi, NBINS).double()
        T = torch.stack([torch.cos(y*5)*0+1, torch.sin(y), torch.cos(2*y), torch.sin(2*y)])
        m = z @ T
        falcon.log({"Signal:mean": m.mean().item()})
        falcon.log({"Signal:std": m.std().item()})
        return m.numpy()

    def get_shape_and_dtype(self):
        return (NBINS,), 'float64'


class Noise:
    """Gaussian noise generator."""
    
    def __init__(self):
        pass
    
    def sample(self, num_samples, parent_conditions=[]):
        result = torch.randn((num_samples, NBINS)).double() * SIGMA
        falcon.log({"Noise:mean": result.mean().item()})
        falcon.log({"Noise:std": result.std().item()})
        return result.numpy()

    def get_shape_and_dtype(self):
        return (NBINS,), 'float64'


class Add:
    """Adds signal and noise components."""
    
    def __init__(self):
        pass
    
    def sample(self, num_samples, parent_conditions=[]):
        m, n = parent_conditions
        m = torch.tensor(m)  # Input is numpy array
        n = torch.tensor(n)  # Input is numpy array
        result = m + n
        falcon.log({"Add:mean": result.mean().item()})
        falcon.log({"Add:std": result.std().item()})
        return result.numpy()

    def get_shape_and_dtype(self):
        return (NBINS,), 'float64'


class E(torch.nn.Module):
    """Embedding network with PCA projection and whitening.
    
    This network implements streaming PCA and whitening for dimensionality
    reduction and preprocessing of high-dimensional observations.
    """
    def __init__(self, log_prefix=None):
        super(E, self).__init__()
        # Import PCA components from their specific modules
        from falcon.contrib.svd import PCAProjector
        from falcon.contrib.norms import DiagonalWhitener
        
        self.projector = PCAProjector(buffer_size=128)
        self.whitener = DiagonalWhitener(NBINS, use_fourier=False)
        self.linear = torch.nn.LazyLinear(NPAR * 2)
        self.log_prefix = log_prefix + ":" if log_prefix else ""

    def forward(self, x, *args):
        falcon.log({f"{self.log_prefix}input_min": x.min().item()})
        falcon.log({f"{self.log_prefix}input_max": x.max().item()})
        if len(args) > 0 and self.training:  # Scaffolds provided
            m, n = args
            # Update whitener with noise component
            self.whitener.update(n+m)
            # Apply whitening to signal component
            white_m = self.whitener(m)
            # Update PCA projector with whitened signal
            self.projector.update(white_m)
        
        # Apply PCA projection to observation
        try:
            #falcon.log({"E:x_input_histogram": wandb.Histogram(x.detach().cpu().flatten().numpy())})
            x = self.whitener(x)
            #falcon.log({"E:x_whitened_histogram": wandb.Histogram(x.detach().cpu().flatten().numpy())})
            x = self.projector(x).float()
            #falcon.log({"E:x_projected_histogram": wandb.Histogram(x.detach().cpu().flatten().numpy())})
        except:
            x = x.float()
        
        # Apply linear transformation
        x = x/NBINS
        x = self.linear(x)
        #falcon.log({"E:x_linear_compression": wandb.Histogram(x.detach().cpu().flatten().numpy())})

        falcon.log({f"{self.log_prefix}output_min": x.min().item()})
        falcon.log({f"{self.log_prefix}output_max": x.max().item()})
        return x
