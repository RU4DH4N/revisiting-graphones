import math

from torch import Tensor, nn

from models.shared.linear import ResidualLinear
from models.shared.validate import require_positive


# https://arxiv.org/abs/2002.05202
def swiglu_hidden_dim(d_model: int, mlp_ratio: float, multiple_of: int) -> int:
    require_positive(d_model=d_model, mlp_ratio=mlp_ratio, multiple_of=multiple_of)
    return multiple_of * math.ceil(round(d_model * mlp_ratio * 2 / 3 / multiple_of, 9))


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, mlp_ratio: float, multiple_of: int) -> None:
        super().__init__()
        hidden = swiglu_hidden_dim(d_model, mlp_ratio, multiple_of)
        self.gate_up = nn.Linear(d_model, 2 * hidden, bias=False)
        self.down_proj = ResidualLinear(hidden, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        gate, up = self.gate_up(x).chunk(2, dim=-1)
        return self.down_proj(nn.functional.silu(gate) * up)
