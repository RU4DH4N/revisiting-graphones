from experiments.arm_names import (
    BASELINE,
    LEARNED,
    LEARNED_DIRECT,
    SPLIT_DIRECT,
    TRIPLE,
    TRIPLE_DIRECT,
)
from pipeline.experiment import Experiment
from workers.records import CKA


class ProjectionAbsorbs(Experiment):
    name = "projection-absorbs"
    primary = (CKA,)
    equivalent = True
    pairs = {
        "learned_vs_triple_with_projection": (LEARNED, TRIPLE),
        "learned_vs_split_with_projection": (LEARNED, BASELINE),
        "learned_vs_triple_without_projection": (LEARNED_DIRECT, TRIPLE_DIRECT),
        "learned_vs_split_without_projection": (LEARNED_DIRECT, SPLIT_DIRECT),
    }
