import torch
import torch.nn

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