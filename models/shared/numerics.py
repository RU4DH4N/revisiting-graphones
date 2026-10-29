import torch
from torch import Tensor

DIVISION_FLOOR = 1e-12
NORM_FLOOR = 1e-6


def normalise(x: Tensor, eps: float) -> Tensor:
    wide = x.float()
    norm = torch.rsqrt(wide.square().sum(dim=-1, keepdim=True) + eps)
    return (wide * norm).to(x.dtype)
