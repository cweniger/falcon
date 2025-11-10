import torch
import torch.nn
import falcon

class Linear(torch.nn.Module):
    def __init__(self, out_features = 64):
        super().__init__()
        self.linear = torch.nn.LazyLinear(out_features)
        self.norm = torch.nn.LazyBatchNorm1d()

    def forward(self, x, y, n):
        # flatten all but first dimension
        x = x.view(x.size(0), -1)
        x = self.norm(x)
        x = self.linear(x)
        return x


class StreamingPCA(torch.nn.Module):
    """Embedding network with PCA projection and whitening.

    This network implements streaming PCA and whitening for dimensionality
    reduction and preprocessing of high-dimensional observations.
    """

    def __init__(self, log_prefix=None, num_bins=10000, num_pars=128):
        super(StreamingPCA, self).__init__()
        # Import PCA components from their specific modules
        from falcon.contrib.svd import PCAProjector
        from falcon.contrib.norms import DiagonalWhitener

        self.projector = PCAProjector(buffer_size=128)
        self.whitener = DiagonalWhitener(num_bins, use_fourier=False)
        self.linear = torch.nn.LazyLinear(num_pars * 2)
        self.log_prefix = log_prefix + ":" if log_prefix else ""
        self.num_bins = num_bins
        self.num_pars = num_pars

    def forward(self, x, m, n):
        falcon.log({f"{self.log_prefix}input_min": x.min().item()})
        falcon.log({f"{self.log_prefix}input_max": x.max().item()})
        if m is not None and self.training:  # Scaffolds provided
            # Update whitener with noise component
            self.whitener.update(n + m)
            # Apply whitening to signal component
            white_m = self.whitener(m)
            # Update PCA projector with whitened signal
            self.projector.update(white_m)

        # Apply PCA projection to observation
        try:
            # falcon.log({"E:x_input_histogram": wandb.Histogram(x.detach().cpu().flatten().numpy())})
            x = self.whitener(x)
            # falcon.log({"E:x_whitened_histogram": wandb.Histogram(x.detach().cpu().flatten().numpy())})
            x = self.projector(x).float()
            # falcon.log({"E:x_projected_histogram": wandb.Histogram(x.detach().cpu().flatten().numpy())})
        except:
            x = x.float()

        # Apply linear transformation
        x = x / self.num_bins
        x = self.linear(x)
        # falcon.log({"E:x_linear_compression": wandb.Histogram(x.detach().cpu().flatten().numpy())})

        falcon.log({f"{self.log_prefix}output_min": x.min().item()})
        falcon.log({f"{self.log_prefix}output_max": x.max().item()})
        return x