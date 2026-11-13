from experiments.arm_names import (
    BYTES_ONLY,
    LEARNED_DIRECT,
    ROTATED_DIRECT,
    ROTATED_HELD,
    TRIPLE,
)
from pipeline.experiment import PREDICTION_AND_GRAMMAR, Experiment


class HeldRotation(Experiment):
    name = "held-rotation"
    primary = PREDICTION_AND_GRAMMAR
    equivalent = False
    pairs = {
        "held_vs_unheld": (ROTATED_HELD, ROTATED_DIRECT),
        "held_vs_projection": (ROTATED_HELD, TRIPLE),
        "held_vs_scaled": (ROTATED_HELD, LEARNED_DIRECT),
        "held_vs_bytes": (ROTATED_HELD, BYTES_ONLY),
    }
