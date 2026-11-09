from experiments.arm_names import (
    LEARNED_DIRECT,
    ROTATED_DIRECT,
    TRIPLE,
    TRIPLE_RANK_MATCHED,
)
from pipeline.experiment import PREDICTION_AND_GRAMMAR, Experiment


class RotationGain(Experiment):
    name = "rotation-gain"
    primary = PREDICTION_AND_GRAMMAR
    equivalent = False
    pairs = {
        "rotated_vs_scaled": (ROTATED_DIRECT, LEARNED_DIRECT),
        "matched_vs_full_projection": (TRIPLE_RANK_MATCHED, TRIPLE),
    }
