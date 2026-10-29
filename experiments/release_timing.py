from experiments.arm_names import FIXED_AT, FIXED_EARLY, FIXED_LATE, LEARNED_DIRECT
from pipeline.experiment import NO_FAMILY, Experiment, against_full


class ReleaseTiming(Experiment):
    name = "release-timing"
    primary = NO_FAMILY
    equivalent = False
    pairs = against_full((FIXED_AT, FIXED_EARLY, FIXED_LATE), LEARNED_DIRECT)
