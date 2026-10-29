from experiments.arm_names import (
    BLIND,
    LEARNED_DIRECT,
    LEARNED_SPLIT_DIRECT,
    NO_HOLD,
    SHUFFLED,
    SOUND_ONLY,
    SPELLING_ONLY,
    SPLIT_DIRECT,
    TRIPLE_DIRECT,
)
from pipeline.experiment import GRAMMAR, Experiment, against_full

OTHERS = (
    NO_HOLD,
    BLIND,
    SHUFFLED,
    SPELLING_ONLY,
    SOUND_ONLY,
    TRIPLE_DIRECT,
    LEARNED_SPLIT_DIRECT,
)


class MixAblation(Experiment):
    name = "mix-ablation"
    primary = GRAMMAR
    equivalent = False
    pairs = {
        **against_full(OTHERS, LEARNED_DIRECT),
        "learned_vs_equal_without_joint": (LEARNED_SPLIT_DIRECT, SPLIT_DIRECT),
    }
