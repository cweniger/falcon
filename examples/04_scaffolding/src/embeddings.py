import torch
import torch.nn
import falcon


class StreamingPCA(torch.nn.Module):
    """Embedding network with PCA projection and whitening.

    This network implements streaming PCA and whitening for dimensionality
    reduction and preprocessing of high-dimensional observations.
    """

    def __init__(self, log_prefix=None, num_bins=10000, num_pars=128, record_every=100):
        super(StreamingPCA, self).__init__()
        # Import PCA components from their specific modules
        from falcon.contrib.svd import PCAProjector
        from falcon.contrib.norms import DiagonalWhitener

        self.projector = PCAProjector(n_components = 32, buffer_size=128)
        self.whitener = DiagonalWhitener(num_bins, use_fourier=False)
        #self.linear = torch.nn.LazyLinear(num_pars * 2)
        self.log_prefix = log_prefix + ":" if log_prefix else ""
        self.num_bins = num_bins
        self.num_pars = num_pars
        self.recorder = falcon.Recorder(every = record_every, base_path = "scaffolding_pca", format='joblib')

        # Simple 1D CNN for feature extraction, with batchnorm and relu activations
#        self.cnn_1d = torch.nn.Sequential(
#            torch.nn.Conv1d(in_channels=1, out_channels=4, kernel_size=5),
#            torch.nn.LazyBatchNorm1d(),
#            torch.nn.ReLU(),
#            torch.nn.MaxPool1d(kernel_size=2),
#            torch.nn.Conv1d(in_channels=4, out_channels=8, kernel_size=5),
#            torch.nn.LazyBatchNorm1d(),
#            torch.nn.ReLU(),
#            torch.nn.MaxPool1d(kernel_size=2),
#            torch.nn.Conv1d(in_channels=8, out_channels=16, kernel_size=5),
#            torch.nn.LazyBatchNorm1d(),
#            torch.nn.ReLU(),
#            torch.nn.Flatten(),
#        )
        self.mlp = torch.nn.Sequential(
            torch.nn.LazyLinear(128),
            torch.nn.ReLU(),
            torch.nn.LazyLinear(256),
            torch.nn.ReLU(),
            torch.nn.LazyLinear(32),
        )

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
            x1 = self.whitener(x)
            m1 = self.whitener(m)
            # falcon.log({"E:x_whitened_histogram": wandb.Histogram(x.detach().cpu().flatten().numpy())})
            x2 = self.projector(x1).float()
            # falcon.log({"E:x_projected_histogram": wandb.Histogram(x.detach().cpu().flatten().numpy())})
            self.recorder(lambda: dict(
                x_input = x[:4].detach().cpu(),
                x_whitened = x1[:4].detach().cpu(),
                m_whitened = m1[:4].detach().cpu(),
                x_projected = x2[:4].detach().cpu(),
            ))
        except:
            x2 = x.float()

        x3 = x2 / self.num_bins**0.5/10
        #print(1, x3.std().item())
        # 1-dim fft, concat real and imag
        x3 = torch.fft.fft(x3, dim=-1)
        #print(x3.std().item())
        x3 = torch.cat([x3.real, x3.imag], dim=-1)
        #print(x3.std().item())
        #x3 = self.cnn_1d(x3.unsqueeze(1))
        x = self.mlp(x3)
        #print(4, x.std().item())
        #x = self.linear(x3)/1e2
        # falcon.log({"E:x_linear_compression": wandb.Histogram(x.detach().cpu().flatten().numpy())})

        falcon.log({f"{self.log_prefix}output_min": x.min().item()})
        falcon.log({f"{self.log_prefix}output_max": x.max().item()})
        return x