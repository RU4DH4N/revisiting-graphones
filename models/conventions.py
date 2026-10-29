from dataclasses import dataclass
from functools import partial


def head_distance(width: int, target: int, heads: int) -> int:
    return abs(width // heads - target)


@dataclass(frozen=True)
class Conventions:
    head_dimensions: int
    width_per_layer: int
    mlp_ratio: float
    multiple_of: int
    norm_eps: float
    rope_theta: float
    initialiser_range: float
    receiver_fraction: float
    message_rank_divisor: int
    gate_hidden: int

    def layers_for(self, width: int) -> int:
        return max(2, round(width / self.width_per_layer))

    def heads_for(self, width: int) -> int:
        fitting = [
            heads
            for heads in range(1, width + 1)
            if width % heads == 0 and width // heads % 2 == 0
        ]
        return min(fitting, key=partial(head_distance, width, self.head_dimensions))

    def shapes(self) -> list[tuple[int, int]]:
        widths = [16, 32, 48, *range(64, 4097, self.head_dimensions)]
        return [
            (width, layers)
            for width in widths
            for layers in range(
                max(2, self.layers_for(width) - 2), self.layers_for(width) + 3
            )
        ]

    def transformer(self, width: int, layers: int, max_seq_len: int) -> dict:
        return {
            "d_model": width,
            "n_layers": layers,
            "n_heads": self.heads_for(width),
            "mlp_ratio": self.mlp_ratio,
            "multiple_of": self.multiple_of,
            "max_seq_len": max_seq_len,
            "rope_theta": self.rope_theta,
            "norm_eps": self.norm_eps,
            "initialiser_range": self.initialiser_range,
        }

    def receiver_rank(self, width: int, message: int) -> int:
        return max(1, min(width, message) // self.message_rank_divisor)
