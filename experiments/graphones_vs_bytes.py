from experiments.arm_names import BASELINE, BYTES_ONLY, LETTERS_ONLY
from pipeline.experiment import Experiment
from workers.records import CKA, MINIMAL_PAIRS


class GraphonesVsBytes(Experiment):
    name = "graphones-vs-bytes"
    primary = (CKA, MINIMAL_PAIRS)
    equivalent = False
    pairs = {
        "split_vs_bytes": (BASELINE, BYTES_ONLY),
        "letters_vs_bytes": (LETTERS_ONLY, BYTES_ONLY),
    }
