import torch
import torch.nn as nn
from math import sqrt
from einops import einsum
from einops import rearrange
from jaxtyping import Float







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

class Embedding(nn.Module):
    def __init__(
            self,
            num_embeddings: int,
            embedding_dim: int,
            device: torch.device = None,
            dtype: torch.dtype = None,
    ):
        super().__init__()
        self.sigma = 1
        self.embedding_dim = embedding_dim
        w = torch.empty(num_embeddings,embedding_dim, device = device, dtype = dtype)
        nn.init.trunc_normal_(w, mean = 0.0, std = self.sigma, a = -3 * self.sigma, b = 3 * self.sigma)
        self.weight = nn.Parameter(w)
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]

class RMSNorm(nn.Module):
    def __init__(
            self,
            d_model: int,
            eps: float = 1e-5,
            device: torch.device = None,
            dtype: torch.dtype = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device = device, dtype = dtype))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32) 
        rms = torch.sqrt(torch.mean(x ** 2, dim = -1, keepdim=True) + self.eps)
        result =  (x / rms) * self.weight
        return result.to(in_dtype)

def SiLU(x):
    return x * torch.sigmoid(x)

class Feedforward(nn.Module):
    def __init__(
            self,
            d_model: int,
            d_ff: int,
            device: torch.device = None,
            dtype: torch.dtype = None,
    ):
        super().__init__()
        self.W_1 = Linear(d_model,d_ff,device,dtype)
        self.W_2 = Linear(d_ff,d_model,device,dtype)
        self.W_3 = Linear(d_model,d_ff,device,dtype)
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.W_2(SiLU(self.W_1(x)) * self.W_3(x))

class RotaryPositionalEmbedding(nn.Module):
    def __init__(
            self,
            theta: float,
            d_k: int,
            max_seq_len: int,
            device: torch.device = None
    ):
        super().__init__()
        position = torch.arange(max_seq_len,device = device,dtype = torch.float32)
        k = torch.arange(1,(d_k/2) + 1,device=device)
        frequency = theta ** (-(2*k - 2)/ d_k)
        rope = einsum(position, frequency, 'i,j -> i j')
        self.register_buffer('rope_cos',torch.cos(rope),persistent = False)
        self.register_buffer('rope_sin',torch.sin(rope),persistent = False)

    def forward(
            self,
            x: torch.Tensor, 
            token_positions: torch.Tensor
    ) -> torch.Tensor:
        cosine = self.rope_cos[token_positions]
        sine = self.rope_sin[token_positions]
        x_1 = x[..., 0:-1:2]
        x_2 = x[..., 1::2]
        y_1 = x_1 * cosine - x_2 * sine
        y_2 = x_1 * sine + x_2 * cosine
        temp = torch.stack([y_1, y_2],dim = -1)
        return rearrange(temp, '... a b -> ... (a b)')

def softmax(x: torch.Tensor, i: int) -> torch.Tensor:
    shifted = torch.exp(x - torch.max(x, dim = i, keepdim = True)[0])
    denomitor = torch.sum(shifted,dim = i,keepdim=True)
    return shifted / denomitor

