import torch
import numpy as np
from numpy.typing import NDArray
import torch.nn as nn
from math import sqrt,cos,pi
from einops import einsum
from einops import rearrange
from jaxtyping import Bool, Float, Int
from torch import Tensor
from collections.abc import Callable, Iterable
from typing import Optional
import os
from typing import BinaryIO,IO


def data_loader(
        x: Int[NDArray,''],
        batch_size,
        context_length,
        device = None,
        dtype = torch.int64
) -> tuple[Tensor,Tensor]:
    B, n, m = batch_size, len(x), context_length
    batch_start = np.random.randint(0,n-m,size = B)
    index = batch_start[:,None] +np.arange(m)
    inputs = torch.tensor(x[index], dtype= dtype,device = device)
    targets = torch.tensor(x[index+1], dtype= dtype,device = device)
    return (inputs, targets)

def save_checkpoint(
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        iteration: int,
        out: str | os.PathLike | BinaryIO | IO[bytes]
):
    state_model = model.state_dict()
    state_opt = optimizer.state_dict()
    obj = {'model': state_model,'opt': state_opt,'iteration': iteration}
    torch.save(obj,out)

def load_checkpoint(
        src: str | os.PathLike | BinaryIO | IO[bytes],
        model: nn.Module,
        optimizer: torch.optim.Optimizer
):
    obj = torch.load(src)
    model.load_state_dict(obj['model'])
    optimizer.load_state_dict(obj['opt'])
    return obj['iteration']

