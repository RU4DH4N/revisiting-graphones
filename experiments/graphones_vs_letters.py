from experiments.arm_names import BASELINE, LETTERS_ONLY
from pipeline.experiment import Experiment
from workers.records import CKA, MINIMAL_PAIRS


class GraphonesVsLetters(Experiment):
    name = "graphones-vs-letters"
    primary = (CKA, MINIMAL_PAIRS)
    equivalent = False
    pairs = {"split_vs_letters": (BASELINE, LETTERS_ONLY)}
