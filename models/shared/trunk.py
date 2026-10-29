import math
from dataclasses import asdict, dataclass

from torch import Tensor, nn

from models.shared.block import Block
from models.shared.linear import ResidualLinear
from models.shared.rope import RotaryEmbedding
from models.shared.validate import require_positive


@dataclass(frozen=True)
class Shape:
    d_model: int
    n_layers: int
    n_heads: int
    mlp_ratio: float
    multiple_of: int
    max_seq_len: int
    rope_theta: float
    norm_eps: float
    initialiser_range: float
    message_dims: int

    def __post_init__(self) -> None:
        require_positive(**asdict(self))
        if self.d_model % self.n_heads:
            raise ValueError(
                f"d_model={self.d_model} not divisible by n_heads={self.n_heads}"
            )
        if self.d_model // self.n_heads % 2:
            raise ValueError(
                f"RoPE needs an even head dimension, d_model={self.d_model}, "
                f"n_heads={self.n_heads}"
            )


def seam(in_dims: int, out_dims: int) -> nn.Module:
    return (
        nn.Identity()
        if in_dims == out_dims
        else nn.Linear(in_dims, out_dims, bias=False)
    )


def initialise(module: nn.Module, std: float, residual_std: float) -> None:
    if isinstance(module, nn.Linear):
        chosen = residual_std if isinstance(module, ResidualLinear) else std
        nn.init.normal_(module.weight, mean=0.0, std=chosen)


# https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
class Trunk(nn.Module):
    def __init__(self, shape: Shape, in_dims: int, out_dims: int) -> None:
        super().__init__()
        require_positive(in_dims=in_dims, out_dims=out_dims)
        self.in_proj = seam(in_dims, shape.d_model)
        rope = RotaryEmbedding(
            shape.d_model // shape.n_heads, shape.max_seq_len, shape.rope_theta
        )
        self.blocks = nn.ModuleList(
            Block(
                shape.d_model,
                shape.n_heads,
                shape.mlp_ratio,
                shape.multiple_of,
                rope,
                shape.norm_eps,
            )
            for _ in range(shape.n_layers)
        )
        self.ln_f = nn.RMSNorm(shape.d_model, eps=shape.norm_eps)
        self.out = seam(shape.d_model, out_dims)

        residual_std = shape.initialiser_range / math.sqrt(2 * shape.n_layers)
        for module in self.modules():
            initialise(module, shape.initialiser_range, residual_std)

    def forward(self, x: Tensor) -> Tensor:
        h = self.in_proj(x)
        for block in self.blocks:
            h = block(h)
        return self.out(self.ln_f(h))
