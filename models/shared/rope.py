import torch
from torch import Tensor, nn


def rotate_half(x: Tensor) -> Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


# https://arxiv.org/abs/2104.09864
class RotaryEmbedding(nn.Module):
    cos: Tensor
    sin: Tensor

    def __init__(self, head_dim: int, max_seq_len: int, theta: float) -> None:
        super().__init__()
        exponent = torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim
        angles = torch.outer(
            torch.arange(max_seq_len, dtype=torch.float32), theta**-exponent
        )
        angles = torch.cat((angles, angles), dim=-1)
        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        steps = x.shape[-2]
        if steps > self.cos.shape[0]:
            raise ValueError(
                f"sequence length {steps} exceeds the configured {self.cos.shape[0]}"
            )
        wide = x.float()
        turned = wide * self.cos[:steps] + rotate_half(wide) * self.sin[:steps]
        return turned.to(x.dtype)
