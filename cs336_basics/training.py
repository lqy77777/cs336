import torch
import torch.nn as nn
from math import sqrt
from einops import einsum
from einops import rearrange
from jaxtyping import Bool, Float, Int
from torch import Tensor
from collections.abc import Callable, Iterable
from typing import Optional


def cross_entropy(
        logits: Float[Tensor,'... vocab_size'],  
        target: Int[Tensor,'...']  #第i+1个位置的真实token编号
) -> float:
    
    max_logits = torch.max(logits,dim = -1, keepdim= True)[0]
    denominator = torch.log(torch.sum(
        torch.exp(logits-max_logits),dim = -1))
    numerator = torch.gather(
        logits, dim = -1,index = target.unsqueeze(-1))
    single = - numerator + max_logits + denominator
    return torch.mean(single)

class AdamW(torch.optim.Optimizer):
    def __init__(self, params,
                  lr = 1e-3,
                  betas = (0.9,0.95),
                  eps = 1e-8,
                  weight_decay = 0.1
    ):
        if lr < 0:
            raise ValueError(f'Invalid learning rate: {lr}')
        defaults = {'lr':lr,
                    'betas': betas,
                    'eps': eps,
                    'weight_decay': weight_decay
        }
        super().__init__(params, defaults)
    
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"] # Get the learning rate.
            beta_1, beta_2 = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p] # Get state associated with p.
                t = state.get("t", 1) # Get iteration number from the state, or 0.
                m = state.get('m', torch.zeros(p.shape,device = p.device,dtype = p.dtype))
                v = state.get('v', torch.zeros(p.shape,device = p.device,dtype = p.dtype))
                grad = p.grad.data # Get the gradient of loss with respect to p.
                alpha = lr * (sqrt(1-beta_2 ** t) / (1-beta_1 ** t))
                m = beta_1 * m + (1-beta_1) * grad
                v = beta_2 * v + (1-beta_2) * (grad ** 2)
                p.data = (1-lr * weight_decay) * p.data
                p.data = p.data - alpha * (m / (torch.sqrt(v) + eps))
                # Update weight tensor in-place.
                state['m'] = m
                state['v'] = v
                state["t"] = t + 1 # Increment iteration number.
        return loss

