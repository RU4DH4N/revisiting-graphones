import math

import torch
from torch import Tensor, nn

from models.shared.linear import ResidualLinear
from models.shared.rope import RotaryEmbedding
from models.shared.swiglu import SwiGLU


def causal_attention(q: Tensor, k: Tensor, v: Tensor) -> Tensor:
    steps = q.shape[-2]
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
    future = torch.ones(steps, steps, dtype=torch.bool, device=q.device).triu(1)
    return torch.softmax(scores.masked_fill(future, -math.inf), dim=-1) @ v


class Block(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        mlp_ratio: float,
        multiple_of: int,
        rope: RotaryEmbedding,
        norm_eps: float,
    ) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.ln1 = nn.RMSNorm(d_model, eps=norm_eps)
        self.ln2 = nn.RMSNorm(d_model, eps=norm_eps)
        self.qkv = nn.Linear(d_model, d_model * 3, bias=False)
        self.out_proj = ResidualLinear(d_model, d_model, bias=False)
        self.mlp = SwiGLU(d_model, mlp_ratio, multiple_of)
        self.rope = rope

    def forward(self, x: Tensor) -> Tensor:
        B, T, C = x.shape
        qkv = self.qkv(self.ln1(x)).reshape(B, T, 3, self.n_heads, self.head_dim)
        stacked = qkv.permute(2, 0, 3, 1, 4)
        q, k = self.rope(stacked[:2])
        v = stacked[2]
        attended = causal_attention(q, k, v)
        x = x + self.out_proj(attended.transpose(1, 2).reshape(B, T, C))
        return x + self.mlp(self.ln2(x))
