from experiments.arm_names import (
    BASELINE,
    BASELINE_MLP,
    LEARNED_DIRECT,
    LEARNED_DIRECT_MATCHED,
    SPLIT_DIRECT,
)
from pipeline.experiment import GRAMMAR, LEARNED_VS_BASELINE, Experiment


class LearnedMix(Experiment):
    name = "learned-mix"
    primary = GRAMMAR
    equivalent = False
    pairs = {
        LEARNED_VS_BASELINE: (LEARNED_DIRECT, BASELINE),
        "matched_learned_vs_baseline": (LEARNED_DIRECT_MATCHED, BASELINE),
        "input_scale": (LEARNED_DIRECT, LEARNED_DIRECT_MATCHED),
        "learned_vs_split_without_projection": (LEARNED_DIRECT, SPLIT_DIRECT),
        "projection_alone": (SPLIT_DIRECT, BASELINE),
        "learned_vs_mlp_projection": (LEARNED_DIRECT, BASELINE_MLP),
    }
