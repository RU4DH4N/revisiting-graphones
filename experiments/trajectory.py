from experiments.arm_names import BASELINE, LEARNED_DIRECT
from pipeline.experiment import LEARNED_VS_BASELINE, NO_FAMILY, Experiment
from pipeline.tasks import Train


class Trajectory(Experiment):
    name = "trajectory"
    primary = NO_FAMILY
    equivalent = False
    pairs = {LEARNED_VS_BASELINE: (LEARNED_DIRECT, BASELINE)}

    def points(self) -> tuple[float, ...]:
        shared = set(Train(self.context, LEARNED_DIRECT).fractions())
        return tuple(sorted(shared & set(Train(self.context, BASELINE).fractions())))
