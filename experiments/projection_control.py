from experiments.arm_names import LETTERS_ONLY, TRIPLE_RANK_MATCHED
from pipeline.experiment import PREDICTION_AND_GRAMMAR, Experiment


class ProjectionControl(Experiment):
    name = "projection-control"
    primary = PREDICTION_AND_GRAMMAR
    equivalent = False
    pairs = {
        "matched_vs_letters": (TRIPLE_RANK_MATCHED, LETTERS_ONLY),
    }
