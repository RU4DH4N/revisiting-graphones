import math
from collections.abc import Callable
from dataclasses import asdict, dataclass
from functools import cache, partial

import torch
from torch import nn

from models.bilingual import WordReceiver
from models.conventions import Conventions
from models.shared.trunk import Shape, Trunk
from workers.settings import Config


def count_parameters(factory: Callable[[], nn.Module]) -> int:
    with torch.device("meta"):
        return sum(p.numel() for p in factory().parameters())


def shape_distance(
    lines: dict[int, tuple[int, int]], target: float, shape: tuple[int, int]
) -> float:
    base, block = lines[shape[0]]
    return abs(math.log((base + (shape[1] - 2) * block) / target))


def closest_shape(
    target: float, count: Callable[[int, int], int], model: Conventions
) -> tuple[int, int]:
    lines = {}
    for width, _ in model.shapes():
        if width not in lines:
            two = count(width, 2)
            lines[width] = (two, count(width, 3) - two)
    return min(model.shapes(), key=partial(shape_distance, lines, target))


def depth_distance(count: Callable[[int], int], target: float, layers: int) -> float:
    return abs(math.log(count(layers) / target))


def closest_depth(
    target: float, width: int, count: Callable[[int], int], model: Conventions
) -> int:
    return min(
        range(1, 4 * model.layers_for(width) + 1),
        key=partial(depth_distance, count, target),
    )


def shape(
    width: int, layers: int, positions: int, message: int, model: Conventions
) -> Shape:
    return Shape(message_dims=message, **model.transformer(width, layers, positions))


class CountedTrunk(nn.Module):
    def __init__(self, shape: Shape, in_dims: int) -> None:
        super().__init__()
        self.trunk = Trunk(shape, in_dims, shape.message_dims)
        self.norm = nn.RMSNorm(shape.message_dims, eps=shape.norm_eps)


def build_receiver(shape: Shape, model: Conventions) -> WordReceiver:
    return WordReceiver(shape, model.receiver_rank(shape.d_model, shape.message_dims))


@cache
def body_count(width: int, layers: int, max_bytes: int, model: Conventions) -> int:
    with torch.device("meta"):
        built = build_receiver(shape(width, layers, max_bytes + 1, width, model), model)
    return sum(
        p.numel()
        for name, p in built.named_parameters()
        if not name.startswith("channels.")
    )


@cache
def trunk_count(
    width: int, layers: int, events: int, message: int, model: Conventions
) -> int:
    return shaped_trunk_count(shape(width, layers, events, message, model), width)


@cache
def shaped_trunk_count(shape: Shape, in_dims: int) -> int:
    return count_parameters(partial(CountedTrunk, shape, in_dims))


@cache
def receiver_body(
    trunk_parameters: int, max_bytes: int, model: Conventions
) -> tuple[int, int]:
    return closest_shape(
        trunk_parameters * model.receiver_fraction,
        partial(body_count, max_bytes=max_bytes, model=model),
        model,
    )


@cache
def counted_receiver(shape: Shape, model: Conventions) -> int:
    return count_parameters(partial(build_receiver, shape, model))


def shaped_receiver_count(
    width: int, layers: int, max_bytes: int, message: int, model: Conventions
) -> int:
    return counted_receiver(shape(width, layers, max_bytes + 1, message, model), model)


@cache
def receiver_layout(
    trunk_parameters: int, max_bytes: int, scale: int, model: Conventions
) -> tuple[int, int, int]:
    width, layers = receiver_body(trunk_parameters, max_bytes, model)
    target = scale * shaped_receiver_count(width, layers, max_bytes, width, model)
    counted = partial(
        shaped_receiver_count, max_bytes=max_bytes, message=width, model=model
    )
    wide, deep = (
        (width, layers) if scale == 1 else closest_shape(target, counted, model)
    )
    return wide, deep, width


def receiver_for(config: Config, max_bytes: int, scale: int) -> Shape:
    width, layers, message = receiver_layout(
        config.trunk_parameters, max_bytes, scale, config.model
    )
    return shape(width, layers, max_bytes + 1, message, config.model)


def receiver_count(config: Config, max_bytes: int, scale: int) -> int:
    return counted_receiver(receiver_for(config, max_bytes, scale), config.model)


def trunk_shape(config: Config, max_bytes: int) -> Shape:
    model = config.model
    events = config.context_words
    message = receiver_for(config, max_bytes, 1).message_dims
    width = config.trunk_width
    counted = partial(trunk_count, width, events=events, message=message, model=model)
    layers = closest_depth(config.trunk_parameters, width, counted, model)
    return shape(width, layers, events, message, model)


def steps(config: Config, train_words: int) -> int:
    return math.ceil(
        config.passes * train_words / (config.batch_windows * config.context_words)
    )


@dataclass(frozen=True)
class Sizing:
    max_unit_bytes: int
    train_words: int
    trunk: Shape
    steps: int


def sized(config: Config, max_bytes: int, words: int, steps: int) -> Sizing:
    return Sizing(max_bytes, words, trunk_shape(config, max_bytes), steps)


def sizing_record(sizing: Sizing) -> dict:
    return {
        "max_unit_bytes": sizing.max_unit_bytes,
        "train_words": sizing.train_words,
        "shapes": {"trunk": asdict(sizing.trunk)},
        "steps": sizing.steps,
    }


def read_sizing(raw: dict) -> Sizing:
    return Sizing(
        raw["max_unit_bytes"],
        raw["train_words"],
        Shape(**raw["shapes"]["trunk"]),
        raw["steps"],
    )
