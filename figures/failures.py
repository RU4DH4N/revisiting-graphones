from pathlib import Path

import numpy as np
from matplotlib.axes import Axes

from experiments.arm_names import BYTES_ONLY, ROTATED_DIRECT, ROTATED_HELD, TRIPLE
from experiments.held_rotation import HeldRotation
from figures.common import MUTED, TEXT_WIDTH, Inputs, saved, smoothed, styled, written
from figures.tables import TABLES, macro, number
from workers.records import BITS_PER_BYTE
from workers.statistics import paired_t

NEEDS = (HeldRotation.name,)
FAILED_RUNS = "failed-runs"
NUMBERS = "failed-runs-numbers"
NORMAL_COLOUR = "#0072B2"
FAILED_COLOUR = "#D55E00"
SETTLED = 0.05


def failed(report: dict, arm: str) -> np.ndarray:
    ends = np.asarray(report["arms"][arm]["per_seed"][BITS_PER_BYTE], dtype=float)
    bytes_ends = np.asarray(
        report["arms"][BYTES_ONLY]["per_seed"][BITS_PER_BYTE], dtype=float
    )
    return ends > bytes_ends.mean()


def normal_lines(prefix: str, report: dict, lost: np.ndarray) -> list[str]:
    lines = []
    for contrast, measures in report["contrasts"].items():
        for measure, cell in measures.items():
            differences = np.asarray(cell["differences"], dtype=float)
            if len(differences) != len(lost):
                continue
            kept = differences[~lost]
            test = paired_t(kept)
            name = macro(prefix, HeldRotation.name, "normal", contrast, measure)
            lines += [
                f"\\newcommand{{\\{name}Mean}}{{{number(test.mean)}}}",
                f"\\newcommand{{\\{name}Low}}{{{number(test.low)}}}",
                f"\\newcommand{{\\{name}High}}{{{number(test.high)}}}",
                f"\\newcommand{{\\{name}Wins}}{{{int((kept > 0).sum())}}}",
                f"\\newcommand{{\\{name}Seeds}}{{{len(kept)}}}",
            ]
    return lines


def curves(inputs: Inputs, arm: str) -> np.ndarray:
    found = [inputs.steps(arm, replicate) for replicate in range(inputs.seeds(arm))]
    width = max(1, len(found[0]) // 100)
    return np.stack([smoothed(loss, width) for loss in found])


def first_step(inputs: Inputs, arm: str, lines: np.ndarray) -> int:
    return len(inputs.steps(arm, 0)) - lines.shape[1]


def separated(lines: np.ndarray, lost: np.ndarray) -> int | None:
    apart = lines[lost].min(axis=0) > lines[~lost].max(axis=0)
    if not apart[-1]:
        return None
    together = np.flatnonzero(~apart)
    return 0 if len(together) == 0 else int(together[-1]) + 1


def separation_lines(
    prefix: str, inputs: Inputs, lines: np.ndarray, lost: np.ndarray
) -> list[str]:
    name = macro(prefix, HeldRotation.name, "separated")
    at = separated(lines, lost) if lost.any() and not lost.all() else None
    if at is None:
        return []
    total = len(inputs.steps(ROTATED_HELD, 0))
    step = at + first_step(inputs, ROTATED_HELD, lines)
    return [
        f"\\newcommand{{\\{name}Step}}{{{step}}}",
        f"\\newcommand{{\\{name}Fraction}}{{{step / total:.2f}}}",
        f"\\newcommand{{\\{name}FailedLoss}}{{{lines[lost, at].min():.3f}}}",
        f"\\newcommand{{\\{name}NormalLoss}}{{{lines[~lost, at].max():.3f}}}",
    ]


def count_lines(prefix: str, report: dict) -> list[str]:
    bytes_ends = report["arms"][BYTES_ONLY]["per_seed"][BITS_PER_BYTE]
    held = macro(prefix, HeldRotation.name, "held")
    unheld = macro(prefix, HeldRotation.name, "unheld")
    mean = macro(prefix, HeldRotation.name, "bytes")
    return [
        f"\\newcommand{{\\{held}Failed}}{{{int(failed(report, ROTATED_HELD).sum())}}}",
        f"\\newcommand{{\\{unheld}Failed}}{{{int(failed(report, ROTATED_DIRECT).sum())}}}",
        f"\\newcommand{{\\{mean}Mean}}{{{float(np.mean(bytes_ends)):.3f}}}",
    ]


def drawn(axis: Axes, start: int, lines: np.ndarray, lost: np.ndarray) -> None:
    steps = start + np.arange(lines.shape[1])
    for replicate, line in enumerate(lines):
        if lost[replicate]:
            axis.plot(steps, line, color=FAILED_COLOUR, linestyle="--", linewidth=1.2)
            axis.annotate(
                f"seed {replicate}",
                (steps[-1], line[-1]),
                xytext=(3, 0),
                textcoords="offset points",
                va="center",
                fontsize=7,
            )
        else:
            axis.plot(steps, line, color=NORMAL_COLOUR, linewidth=0.7, alpha=0.8)


def legend(axis: Axes, lost: np.ndarray) -> None:
    axis.plot([], [], color=MUTED, linewidth=0.7, label="projection")
    axis.plot([], [], color=NORMAL_COLOUR, linewidth=0.7, label="held gate")
    if lost.any():
        axis.plot(
            [],
            [],
            color=FAILED_COLOUR,
            linestyle="--",
            linewidth=1.2,
            label="held gate, ends worse than the average bytes run",
        )
    axis.legend(loc="upper right")


def failed_runs(inputs: Inputs) -> list[Path]:
    report = inputs.report(HeldRotation.name)
    prefix = inputs.config_path.stem
    lost = failed(report, ROTATED_HELD)
    held = curves(inputs, ROTATED_HELD)
    projection = curves(inputs, TRIPLE)
    start = first_step(inputs, ROTATED_HELD, held)
    figure, axes = styled(1, 1, TEXT_WIDTH, 2.4)
    axis = axes[0, 0]
    steps = start + np.arange(projection.shape[1])
    for line in projection:
        axis.plot(steps, line, color=MUTED, linewidth=0.7, alpha=0.6)
    drawn(axis, start, held, lost)
    legend(axis, lost)
    settled = int(SETTLED * held.shape[1])
    both = np.concatenate([held, projection])
    axis.set_ylim(both.min() * 0.98, both[:, settled].max())
    axis.set_xlim(0, (start + held.shape[1]) * 1.08)
    axis.set_xlabel("training step")
    axis.set_ylabel("training loss")
    lines = [
        *count_lines(prefix, report),
        *normal_lines(prefix, report, lost),
        *separation_lines(prefix, inputs, held, lost),
    ]
    return [
        saved(inputs, figure, FAILED_RUNS),
        written(inputs.folder / TABLES / f"{NUMBERS}.tex", "\n".join(lines) + "\n"),
    ]
