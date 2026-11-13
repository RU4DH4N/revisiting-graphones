from pathlib import Path

from experiments.fixed_tables import FixedTables
from experiments.graphones_vs_bytes import GraphonesVsBytes
from experiments.graphones_vs_letters import GraphonesVsLetters
from experiments.headline import Headline
from experiments.held_rotation import HeldRotation
from experiments.learned_mix import LearnedMix
from experiments.letter_cut import LetterCut
from experiments.mix_ablation import MixAblation
from experiments.one_language import OneLanguage
from experiments.projection_absorbs import ProjectionAbsorbs
from experiments.receiver_size import ReceiverSize
from experiments.release_timing import ReleaseTiming
from experiments.rotation_gain import RotationGain
from experiments.rotation_matches_projection import RotationMatchesProjection
from experiments.trajectory import Trajectory
from framework.paths import portable_path
from framework.storage import write_json
from orchestrator.contract import CPU, LOCAL, Contract
from pipeline.tasks import Task

EXPERIMENTS = (
    Headline,
    GraphonesVsLetters,
    LearnedMix,
    ProjectionAbsorbs,
    MixAblation,
    FixedTables,
    ReceiverSize,
    ReleaseTiming,
    Trajectory,
    LetterCut,
    GraphonesVsBytes,
    RotationMatchesProjection,
    RotationGain,
)
REPORTED = (*EXPERIMENTS, OneLanguage, HeldRotation)


class Everything(Task):
    gpu = CPU
    placement = LOCAL
    module = "experiments.everything"

    def configuration(self) -> dict:
        return self.context.identity

    def task_parameters(self) -> dict:
        return {"split": self.context.split}

    def dependencies(self) -> tuple[Contract, ...]:
        return tuple(experiment(self.context) for experiment in EXPERIMENTS)

    def outputs(self) -> tuple[Path, ...]:
        return (self.layout.experiment("index", self.context.split),)

    def run(self) -> None:
        write_json(
            self.outputs()[0],
            {
                dependency.label: [portable_path(p) for p in dependency.outputs()]
                for dependency in self.dependencies()
            },
        )


BY_FILE: dict[str, type[Task]] = {
    "everything": Everything,
    "headline": Headline,
    "graphones_vs_letters": GraphonesVsLetters,
    "projection_absorbs": ProjectionAbsorbs,
    "learned_mix": LearnedMix,
    "mix_ablation": MixAblation,
    "fixed_tables": FixedTables,
    "receiver_size": ReceiverSize,
    "release_timing": ReleaseTiming,
    "trajectory": Trajectory,
    "letter_cut": LetterCut,
    "graphones_vs_bytes": GraphonesVsBytes,
    "rotation_matches_projection": RotationMatchesProjection,
    "rotation_gain": RotationGain,
    "one_language": OneLanguage,
    "held_rotation": HeldRotation,
}
