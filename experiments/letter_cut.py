from experiments.arm_names import KEPT_LETTERS, LEARNED_DIRECT, TRIPLE_DIRECT
from pipeline.experiment import Experiment
from workers.records import CKA, POS_TRANSFER


class LetterCut(Experiment):
    name = "letter-cut"
    primary = (CKA, POS_TRANSFER)
    equivalent = False
    pairs = {
        "kept_vs_learned": (KEPT_LETTERS, LEARNED_DIRECT),
        "kept_vs_triple": (KEPT_LETTERS, TRIPLE_DIRECT),
    }
