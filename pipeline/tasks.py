import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

from data.bilingual import CORPUS_FILES
from data.graphones import (
    ALIGNMENT_FILE,
    FILES as GRAPHONE_FILES,
    code_files,
    table_file,
)
from framework.checkpoints import sidecar_path
from framework.paths import ROOT, portable_path
from framework.sources import closure
from orchestrator.context import Context
from orchestrator.contract import CPU, LOCAL, SHARED, Contract, estimate_of
from workers import entry, estimates
from workers.layout import Layout
from workers.settings import Arm, Config

DESIGN_FIELDS = (
    "data",
    "model",
    "trunk_parameters",
    "trunk_width",
    "context_words",
    "batch_windows",
    "passes",
)
TRAINING_FIELDS = (
    *DESIGN_FIELDS,
    "root_seed",
    "learning_rate",
    "gate_rate",
    "optimiser",
    "hold",
    "precision",
    "log_interval",
    "snapshot_interval",
    "gauge",
)


class Task(Contract):
    gpu = SHARED
    module = ""
    fields: tuple[str, ...] = TRAINING_FIELDS

    def __init__(self, context: Context) -> None:
        self.context = context

    @property
    def layout(self) -> Layout:
        return Layout(self.context.output)

    @property
    def settings(self) -> Config:
        return self.context.config

    def configuration(self) -> dict:
        return {field: self.context.identity[field] for field in self.fields}

    def parameters(self) -> dict:
        return {
            "config": self.configuration(),
            "output": portable_path(self.context.output),
            **self.task_parameters(),
        }

    def task_parameters(self) -> dict:
        return {}

    def sources(self) -> tuple[Path, ...]:
        return tuple(sorted({*super().sources(), *closure(self.module)}))

    def worker_arguments(self) -> tuple[str, ...]:
        return ()

    def command(self) -> list[str]:
        return [
            "-m",
            self.module,
            *self.worker_arguments(),
            entry.CONFIG,
            self.context.config_argument,
            entry.OUTPUT,
            self.context.output_argument,
            entry.DEVICE,
            self.context.device,
        ]

    def run(self) -> None:
        subprocess.run([sys.executable, *self.command()], cwd=ROOT, check=True)


class Prepare(Task):
    gpu = CPU
    placement = LOCAL
    module = "workers.run_prepare"
    fields = ("data",)

    def estimate(self) -> dict:
        return estimate_of("prepare", 0.0, estimates.corpus_bytes(self.settings))

    def outputs(self) -> tuple[Path, ...]:
        return tuple(self.layout.corpus / name for name in CORPUS_FILES)


class Design(Task):
    gpu = CPU
    placement = LOCAL
    module = "workers.run_design"
    fields = DESIGN_FIELDS

    def estimate(self) -> dict:
        return estimate_of("design", 0.0, 0)

    def dependencies(self) -> tuple[Contract, ...]:
        return (Prepare(self.context),)

    def outputs(self) -> tuple[Path, ...]:
        return (self.layout.design,)


class Graphones(Task):
    gpu = CPU
    placement = LOCAL
    module = "workers.run_graphones"
    fields = ("graphones", "root_seed")

    def configuration(self) -> dict:
        graphones = self.settings.graphones
        voices = {
            language: graphones.voices[language]
            for language in self.settings.data.languages
        }
        return {
            **super().configuration(),
            "graphones": {**asdict(graphones), "voices": voices},
            "code_dims": self.settings.code_widths(),
        }

    def estimate(self) -> dict:
        return estimate_of("graphones", 0.0, estimates.GRAPHONE_BYTES)

    def dependencies(self) -> tuple[Contract, ...]:
        return (Prepare(self.context),)

    def outputs(self) -> tuple[Path, ...]:
        root = self.layout.graphones
        names = [
            *GRAPHONE_FILES,
            ALIGNMENT_FILE,
            *(
                name
                for dims in self.settings.code_widths()
                for name in code_files(dims)
            ),
        ]
        return tuple(root / name for name in names)


class MixTables(Task):
    gpu = CPU
    module = "workers.run_tables"
    fields = ("tables",)

    def configuration(self) -> dict:
        return {**super().configuration(), "names": self.settings.table_names()}

    def estimate(self) -> dict:
        return estimate_of("tables", 0.0, 0)

    def dependencies(self) -> tuple[Contract, ...]:
        source = self.settings.tables.source
        return (Graphones(self.context), Train(self.context, source))

    def outputs(self) -> tuple[Path, ...]:
        names = self.settings.table_names()
        return tuple(self.layout.graphones / table_file(name) for name in names)


class ArmTask(Task):
    def __init__(self, context: Context, arm: str) -> None:
        super().__init__(context)
        if arm not in context.config.arms:
            raise ValueError(f"the config defines no arm named {arm}")
        if context.config.arms[arm].ruled and len(context.config.data.languages) < 2:
            raise ValueError(
                f"arm {arm} releases its gate on the language gauge, "
                "which needs two languages"
            )
        self.arm = arm

    @property
    def spec(self) -> Arm:
        return self.settings.arms[self.arm]

    def task_parameters(self) -> dict:
        return {"arm": self.arm, "spec": self.context.identity["arms"][self.arm]}

    def worker_arguments(self) -> tuple[str, ...]:
        return (entry.ARM, self.arm)

    def dependencies(self) -> tuple[Contract, ...]:
        shared = (Prepare(self.context), Design(self.context), Graphones(self.context))
        tabled = (MixTables(self.context),) if self.spec.table is not None else ()
        return (*shared, *tabled)


class Train(ArmTask):
    module = "workers.run_train"

    @property
    def label(self) -> str:
        return f"Train[{self.arm}]"

    def fractions(self) -> tuple[float, ...]:
        marked = self.settings.checkpoints
        return marked.fractions if self.arm in marked.arms else ()

    def configuration(self) -> dict:
        return {**super().configuration(), "checkpoints": list(self.fractions())}

    def dependencies(self) -> tuple[Contract, ...]:
        source = self.spec.release_from
        released = () if source is None else (Releases(self.context, source),)
        return (*super().dependencies(), *released)

    def estimate(self) -> dict:
        seeds = self.spec.seeds
        saved = 1 + len(self.fractions())
        return estimate_of(
            "train",
            seeds * estimates.train_flops(self.settings, self.spec),
            seeds * saved * estimates.checkpoint_bytes(self.settings, self.spec),
        )

    def outputs(self) -> tuple[Path, ...]:
        found: list[Path] = []
        for replicate in range(self.spec.seeds):
            checkpoint = self.layout.checkpoint(self.arm, replicate)
            found += [
                checkpoint,
                sidecar_path(checkpoint),
                self.layout.metrics(self.arm, replicate),
                self.layout.gates(self.arm, replicate),
                self.layout.steps(self.arm, replicate),
            ]
            for fraction in self.fractions():
                saved = self.layout.checkpoint_at(self.arm, replicate, fraction)
                found += [saved, sidecar_path(saved)]
        return tuple(found)


class Releases(ArmTask):
    gpu = CPU
    placement = LOCAL
    module = "workers.run_releases"

    @property
    def label(self) -> str:
        return f"Releases[{self.arm}]"

    def estimate(self) -> dict:
        return estimate_of("releases", 0.0, 0)

    def dependencies(self) -> tuple[Contract, ...]:
        return (Train(self.context, self.arm),)

    def outputs(self) -> tuple[Path, ...]:
        return (self.layout.releases(self.arm),)


class Evaluate(ArmTask):
    module = "workers.run_evaluate"
    fields = (*TRAINING_FIELDS, "evaluation")

    def __init__(
        self, context: Context, arm: str, replicate: int, at: float | None
    ) -> None:
        super().__init__(context, arm)
        if not 0 <= replicate < self.spec.seeds:
            raise ValueError(f"arm {arm} has no seed {replicate}")
        if at is not None and at not in Train(context, arm).fractions():
            raise ValueError(f"arm {arm} saves no checkpoint at {at}")
        self.replicate = replicate
        self.at = at

    def task_parameters(self) -> dict:
        return {
            **super().task_parameters(),
            "replicate": self.replicate,
            "split": self.context.split,
            "at": self.at,
        }

    @property
    def label(self) -> str:
        point = "" if self.at is None else f"@{self.at}"
        return f"Evaluate[{self.arm}/{self.replicate}/{self.context.split}{point}]"

    def estimate(self) -> dict:
        split = self.context.split
        return estimate_of(
            "evaluate",
            estimates.evaluate_flops(self.settings, self.spec, split),
            estimates.scores_bytes(self.settings, split),
        )

    def dependencies(self) -> tuple[Contract, ...]:
        return (*super().dependencies(), Train(self.context, self.arm))

    def outputs(self) -> tuple[Path, ...]:
        split, layout = self.context.split, self.layout
        if self.at is None:
            return (
                layout.evaluation(self.arm, self.replicate, split),
                layout.scores(self.arm, self.replicate, split),
            )
        return (
            layout.evaluation_at(self.arm, self.replicate, split, self.at),
            layout.scores_at(self.arm, self.replicate, split, self.at),
        )

    def worker_arguments(self) -> tuple[str, ...]:
        point = () if self.at is None else (entry.AT, str(self.at))
        return (
            *super().worker_arguments(),
            entry.REPLICATE,
            str(self.replicate),
            entry.SPLIT,
            self.context.split,
            *point,
        )


def evaluations(context: Context, arm: str) -> tuple[Evaluate, ...]:
    return tuple(
        Evaluate(context, arm, replicate, None)
        for replicate in range(context.config.arms[arm].seeds)
    )


def checkpoint_evaluations(context: Context, arm: str) -> tuple[Evaluate, ...]:
    return tuple(
        Evaluate(context, arm, replicate, fraction)
        for fraction in Train(context, arm).fractions()
        for replicate in range(context.config.arms[arm].seeds)
    )
