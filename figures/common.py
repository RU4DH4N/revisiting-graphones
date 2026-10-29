import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from data.graphones import Tables, read_tables
from experiments.arm_names import (
    BASELINE,
    FIXED_AT,
    FIXED_EARLY,
    FIXED_LATE,
    LEARNED_DIRECT,
    NO_HOLD,
)
from framework.paths import portable_path
from framework.storage import read_json
from workers.analysis import LANGUAGE_LATE
from workers.layout import Layout
from workers.records import (
    BITS_PER_BYTE,
    CKA,
    MINIMAL_PAIRS,
    POS_ACCURACY,
    POS_SELECTIVITY,
    POS_TRANSFER,
    RETRIEVAL,
    RETRIEVAL_CSLS,
    GateRecord,
    Metrics,
    read_gates,
    read_metrics,
)
from workers.settings import Config

INK = "#0b0b0b"
MUTED = "#52514e"
RULE = "#d8d7d2"
TEXT_WIDTH = 4.8
# https://doi.org/10.1038/nmeth.1618
COLOURS = {
    LEARNED_DIRECT: "#0072B2",
    BASELINE: "#D55E00",
    FIXED_AT: "#009E73",
    FIXED_EARLY: "#E69F00",
    FIXED_LATE: "#CC79A7",
    NO_HOLD: "#56B4E9",
}
LINESTYLES = {
    LEARNED_DIRECT: "-",
    BASELINE: "--",
    FIXED_AT: "-.",
    FIXED_EARLY: ":",
    FIXED_LATE: (0, (5, 1, 1, 1)),
    NO_HOLD: (0, (1, 2)),
}
MARKERS = {
    LEARNED_DIRECT: "o",
    BASELINE: "s",
    FIXED_AT: "D",
    FIXED_EARLY: "^",
    FIXED_LATE: "v",
    NO_HOLD: "P",
}
STREAM_COLOURS = ("#0072B2", "#D55E00", "#009E73")
STREAM_LINESTYLES = ("-", "--", ":")
STREAM_MARKERS = ("o", "s", "^")
STREAM_NAMES = ("letter", "sound", "joint")
LABELS = {
    BITS_PER_BYTE: "bits per byte",
    MINIMAL_PAIRS: "MultiBLiMP accuracy",
    POS_ACCURACY: "POS accuracy",
    POS_SELECTIVITY: "POS selectivity",
    POS_TRANSFER: "POS transfer",
    RETRIEVAL: "retrieval P@1",
    RETRIEVAL_CSLS: "retrieval CSLS P@1",
    CKA: "linear CKA",
    LANGUAGE_LATE: "late language share",
}


@dataclass(frozen=True)
class Inputs:
    config: Config
    config_path: Path
    layout: Layout
    split: str

    @property
    def folder(self) -> Path:
        return self.layout.root / "figures" / self.split

    def report_path(self, experiment: str) -> Path:
        return self.layout.experiment(experiment, self.split)

    def report(self, experiment: str) -> dict:
        return read_json(self.report_path(experiment))

    def training(self, arm: str, replicate: int) -> Metrics:
        return read_metrics(read_json(self.layout.metrics(arm, replicate)))

    def gates(self, arm: str, replicate: int) -> GateRecord:
        return read_gates(self.layout.gates(arm, replicate))

    def steps(self, arm: str, replicate: int) -> np.ndarray:
        return np.load(self.layout.steps(arm, replicate))["loss"]

    def scores(self, arm: str, replicate: int) -> dict[str, np.ndarray]:
        found = np.load(self.layout.scores(arm, replicate, self.split))
        return {name: found[name] for name in found}

    def tables(self) -> Tables:
        return read_tables(self.layout.graphones)

    def seeds(self, arm: str) -> int:
        return self.config.arms[arm].seeds


def missing(inputs: Inputs, experiments: tuple[str, ...]) -> list[str]:
    lines = []
    for name in experiments:
        if not inputs.report_path(name).is_file():
            stem = name.replace("-", "_")
            lines.append(
                f"missing {portable_path(inputs.report_path(name))}; produce it with: "
                f"uv run -m orchestrator experiments/{stem}.py --config "
                f"{portable_path(inputs.config_path)} --output "
                f"{portable_path(inputs.layout.root)} --split {inputs.split} "
                "--device cpu --run"
            )
    return lines


def styled(
    rows: int, columns: int, width: float, height: float
) -> tuple[Figure, np.ndarray]:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["cmr10"],
            "mathtext.fontset": "cm",
            "axes.formatter.use_mathtext": True,
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.6,
            "axes.edgecolor": MUTED,
            "axes.labelcolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": RULE,
            "grid.linewidth": 0.5,
            "legend.frameon": False,
            "lines.linewidth": 1.2,
            "lines.markersize": 4,
        }
    )
    figure, axes = plt.subplots(
        rows, columns, figsize=(width, height), squeeze=False, layout="constrained"
    )
    return figure, axes


def saved(inputs: Inputs, figure: Figure, name: str) -> Path:
    inputs.folder.mkdir(parents=True, exist_ok=True)
    path = inputs.folder / f"{name}.pdf"
    figure.savefig(path)
    plt.close(figure)
    return path


def panel(axis: Axes, index: int) -> None:
    axis.text(
        -0.02,
        1.02,
        f"({'abcdefgh'[index]})",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
    )


def zero_line(axis: Axes, vertical: bool) -> None:
    if vertical:
        axis.axvline(0.0, color=MUTED, linewidth=0.8)
    else:
        axis.axhline(0.0, color=MUTED, linewidth=0.8)


def label(measure: str) -> str:
    return LABELS.get(measure, measure)


def smoothed(values: np.ndarray, width: int) -> np.ndarray:
    if width <= 1 or len(values) < width:
        return values
    kernel = np.ones(width) / width
    return np.convolve(values, kernel, mode="valid")


def written(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def dumped(path: Path, value: dict) -> Path:
    return written(path, json.dumps(value, indent=1))
