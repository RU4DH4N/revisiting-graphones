from dataclasses import dataclass

from data.splits import TRAIN
from models.shared.trunk import Shape
from workers import design
from workers.settings import Arm, Config

BYTES_PER_WORD = 6.2
TYPES_PER_WORD = 0.08
UNIT_BYTES = 24
BENCHMARK_WORDS = 450_000
BYTES_PER_PARAMETER = 4
TRAINING_FLOPS_PER_PARAMETER = 6
FORWARD_FLOPS_PER_PARAMETER = 2
BYTES_PER_ID = 4
SCORE_BYTES_PER_WORD = 7
GRAPHONE_BYTES = 260_000_000


@dataclass(frozen=True)
class Plan:
    words: dict[str, float]
    total_words: float
    steps: int
    trunk: Shape


PLANS: dict[int, tuple[Config, Plan]] = {}


def planned(config: Config) -> Plan:
    languages = len(config.data.languages)
    words = {
        split: size * languages / BYTES_PER_WORD
        for split, size in config.data.bytes.items()
    }
    return Plan(
        words,
        sum(words.values()) + BENCHMARK_WORDS,
        design.steps(config, int(words[TRAIN])),
        design.trunk_shape(config, UNIT_BYTES),
    )


def plan(config: Config) -> Plan:
    if id(config) not in PLANS:
        PLANS[id(config)] = (config, planned(config))
    return PLANS[id(config)][1]


def mlp_extra(config: Config, arm: Arm) -> int:
    if not arm.input_mlp:
        return 0
    shape = plan(config).trunk
    hidden = round(shape.mlp_ratio * shape.d_model)
    return (
        arm.code_width * hidden
        + hidden * shape.d_model
        - arm.code_width * shape.d_model
    )


def rank_saving(config: Config, arm: Arm) -> int:
    if arm.projection_rank is None:
        return 0
    width = plan(config).trunk.d_model
    return arm.code_width * width - arm.projection_rank * (arm.code_width + width)


def trunk_parameters(config: Config, arm: Arm) -> int:
    return (
        design.shaped_trunk_count(plan(config).trunk, arm.code_width)
        + mlp_extra(config, arm)
        - rank_saving(config, arm)
    )


def parameters(config: Config, arm: Arm) -> int:
    return trunk_parameters(config, arm) + design.receiver_count(
        config, UNIT_BYTES, arm.receiver_scale
    )


def flops_per_word(config: Config, arm: Arm) -> float:
    receiver = design.receiver_count(config, UNIT_BYTES, arm.receiver_scale)
    return trunk_parameters(config, arm) + receiver * UNIT_BYTES


def train_flops(config: Config, arm: Arm) -> float:
    return (
        TRAINING_FLOPS_PER_PARAMETER
        * plan(config).steps
        * config.batch_windows
        * config.context_words
        * flops_per_word(config, arm)
    )


def evaluate_flops(config: Config, arm: Arm, split: str) -> float:
    words = plan(config).words[split] + BENCHMARK_WORDS
    return FORWARD_FLOPS_PER_PARAMETER * words * flops_per_word(config, arm)


def checkpoint_bytes(config: Config, arm: Arm) -> int:
    return BYTES_PER_PARAMETER * parameters(config, arm)


def corpus_bytes(config: Config) -> int:
    total = plan(config).total_words
    return int(BYTES_PER_ID * total + total * TYPES_PER_WORD * (UNIT_BYTES + 1))


def scores_bytes(config: Config, split: str) -> int:
    return int(plan(config).words[split] * SCORE_BYTES_PER_WORD)
