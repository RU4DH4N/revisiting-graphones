from experiments.arm_names import BASELINE, LEARNED_DIRECT
from pipeline.experiment import LEARNED_VS_BASELINE, NO_FAMILY, Experiment


class Headline(Experiment):
    name = "headline"
    primary = NO_FAMILY
    equivalent = False
    pairs = {LEARNED_VS_BASELINE: (LEARNED_DIRECT, BASELINE)}
