import math
import random
from contextlib import AbstractContextManager, nullcontext
from typing import Any

import numpy as np
import torch
from torch import Tensor

CUDA = "cuda"
AUTO = "auto"
FP32 = "fp32"
BF16 = "bf16"
PRECISIONS = (AUTO, FP32, BF16)


def on_cuda(device: torch.device) -> bool:
    return device.type == CUDA


def autocast_dtype(device: torch.device, precision: str) -> torch.dtype:
    selected = BF16 if precision == AUTO and on_cuda(device) else precision
    if selected not in PRECISIONS:
        raise ValueError(f"unknown precision={precision!r}")
    if selected == BF16 and not on_cuda(device):
        raise ValueError(
            f"precision={selected!r} requires CUDA, device={device.type!r}"
        )
    return torch.bfloat16 if selected == BF16 else torch.float32


def autocast_context(
    device: torch.device,
    precision: str,
) -> AbstractContextManager[Any]:
    full = autocast_dtype(device, precision) is torch.float32
    return (
        nullcontext()
        if full
        else torch.autocast(device_type=CUDA, dtype=torch.bfloat16)
    )


def adamw(
    decayed: list[Tensor],
    plain: list[Tensor],
    lr: float,
    weight_decay: float,
    betas: tuple[float, float],
    eps: float,
    device: torch.device,
) -> torch.optim.AdamW:
    groups = [
        {"params": decayed, "weight_decay": weight_decay},
        {"params": plain, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(
        [group for group in groups if group["params"]],
        lr=lr,
        betas=betas,
        eps=eps,
        fused=on_cuda(device),
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def cosine_schedule(step: int, steps: int, warmup: int) -> float:
    warming = bool(warmup) and step < warmup
    progress = (step - warmup) / max(steps - warmup - 1, 1)
    decayed = 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
    return (step + 1) / warmup if warming else decayed


def device_name(device: torch.device) -> str:
    return torch.cuda.get_device_name(device) if on_cuda(device) else device.type
