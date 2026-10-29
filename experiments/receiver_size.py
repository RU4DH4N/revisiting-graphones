from experiments.arm_names import (
    BASELINE,
    BASELINE_LARGE,
    BASELINE_SMALL,
    LEARNED_DIRECT,
    LEARNED_DIRECT_LARGE,
    LEARNED_DIRECT_SMALL,
)
from pipeline.experiment import GRAMMAR, Experiment


class ReceiverSize(Experiment):
    name = "receiver-size"
    primary = GRAMMAR
    equivalent = False
    pairs = {
        "small": (LEARNED_DIRECT_SMALL, BASELINE_SMALL),
        "base": (LEARNED_DIRECT, BASELINE),
        "large": (LEARNED_DIRECT_LARGE, BASELINE_LARGE),
    }
