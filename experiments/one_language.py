from experiments.arm_names import BYTES_ONLY, LETTERS_ONLY, ROTATED_DIRECT, TRIPLE
from pipeline.experiment import PREDICTION_AND_GRAMMAR, Experiment


class OneLanguage(Experiment):
    name = "one-language"
    primary = PREDICTION_AND_GRAMMAR
    equivalent = False
    pairs = {
        "graphones_vs_bytes": (TRIPLE, BYTES_ONLY),
        "graphones_vs_letters": (TRIPLE, LETTERS_ONLY),
        "rotated_vs_bytes": (ROTATED_DIRECT, BYTES_ONLY),
    }
