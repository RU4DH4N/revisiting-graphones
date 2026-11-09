from pathlib import Path

from framework.storage import write_json
from orchestrator.contract import CPU, LOCAL, Contract
from pipeline.tasks import Task, Train, checkpoint_evaluations, evaluations
from workers.analysis import (
    ArmResults,
    arm_report,
    contrast_report,
    contrasts,
    family,
)
from workers.records import BITS_PER_BYTE, MINIMAL_PAIRS, POS_SELECTIVITY

GRAMMAR = (MINIMAL_PAIRS, POS_SELECTIVITY)
PREDICTION_AND_GRAMMAR = (BITS_PER_BYTE, *GRAMMAR)
NO_FAMILY: tuple[str, ...] = ()
LEARNED_VS_BASELINE = "learned_vs_baseline"


def against_full(others: tuple[str, ...], full: str) -> dict[str, tuple[str, str]]:
    return {f"{name}_vs_full": (name, full) for name in others}


class Experiment(Task):
    gpu = CPU
    placement = LOCAL
    module = "workers.analysis"
    name: str
    pairs: dict[str, tuple[str, str]]
    primary: tuple[str, ...]
    equivalent: bool

    @property
    def label(self) -> str:
        return f"Experiment[{self.name}]"

    def configuration(self) -> dict:
        return self.context.identity

    def task_parameters(self) -> dict:
        return {"experiment": self.name, "split": self.context.split}

    def arm_names(self) -> tuple[str, ...]:
        names = tuple(
            dict.fromkeys(arm for pair in self.pairs.values() for arm in pair)
        )
        missing = [name for name in names if name not in self.settings.arms]
        if missing:
            raise ValueError(f"{self.name} needs arms the config lacks: {missing}")
        return names

    def points(self) -> tuple[float, ...]:
        return ()

    def dependencies(self) -> tuple[Contract, ...]:
        checkpointed = tuple(
            task
            for arm in self.arm_names()
            for task in checkpoint_evaluations(self.context, arm)
            if task.at in self.points()
        )
        return (
            *(
                task
                for arm in self.arm_names()
                for task in (*evaluations(self.context, arm), Train(self.context, arm))
            ),
            *checkpointed,
        )

    def outputs(self) -> tuple[Path, ...]:
        return (self.layout.experiment(self.name, self.context.split),)

    def results(self, at: float | None) -> dict[str, ArmResults]:
        return {
            name: ArmResults(self.layout, self.settings, name, self.context.split, at)
            for name in self.arm_names()
        }

    def report(self, at: float | None) -> dict:
        arms = self.results(at)
        found = contrasts(arms, self.pairs)
        return {
            "arms": {name: arm_report(arm) for name, arm in arms.items()},
            "contrasts": contrast_report(found, self.primary, self.equivalent),
            "family": family(found, self.primary, self.equivalent),
        }

    def run(self) -> None:
        arms = self.results(None)
        write_json(
            self.outputs()[0],
            {
                "experiment": self.name,
                "split": self.context.split,
                **self.report(None),
                "released_at": {
                    name: arm.released()
                    for name, arm in arms.items()
                    if self.settings.arms[name].hold
                },
                "checkpoints": {str(at): self.report(at) for at in self.points()},
            },
        )
