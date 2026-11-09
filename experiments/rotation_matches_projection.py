from experiments.arm_names import ROTATED_DIRECT, TRIPLE, TRIPLE_RANK_MATCHED
from pipeline.experiment import PREDICTION_AND_GRAMMAR, Experiment


class RotationMatchesProjection(Experiment):
    name = "rotation-matches-projection"
    primary = PREDICTION_AND_GRAMMAR
    equivalent = True
    pairs = {
        "rotated_vs_projection": (ROTATED_DIRECT, TRIPLE),
        "rotated_vs_matched_projection": (ROTATED_DIRECT, TRIPLE_RANK_MATCHED),
    }
