import torch
import torch.nn as nn
from math import sqrt
from einops import einsum
from einops import rearrange








class Linear(nn.Module):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            device: torch.device = None,
            dtype: torch.dtype = None,
    ):
        super().__init__()
        self.sigma = sqrt(2 / (out_features + in_features))
        w = torch.empty(out_features,in_features, device = device, dtype = dtype)
        nn.init.trunc_normal_(w, mean = 0.0, std = self.sigma, a = -3 * self.sigma, b = 3 * self.sigma)
        self.weight = nn.Parameter(w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight.T
