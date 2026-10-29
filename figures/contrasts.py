from pathlib import Path

import numpy as np
from matplotlib.axes import Axes
from matplotlib.ticker import MaxNLocator

from experiments import everything
from experiments.arm_names import (
    FIXED_AT,
    FIXED_EARLY,
    FIXED_LATE,
    LEARNED_DIRECT,
    NO_HOLD,
)
from experiments.mix_ablation import MixAblation
from experiments.release_timing import ReleaseTiming
from experiments.trajectory import Trajectory
from figures.common import (
    COLOURS,
    INK,
    LINESTYLES,
    MARKERS,
    MUTED,
    TEXT_WIDTH,
    Inputs,
    label,
    panel,
    saved,
    styled,
    zero_line,
)
from pipeline.experiment import GRAMMAR, LEARNED_VS_BASELINE

EXPERIMENTS = tuple(experiment.name for experiment in everything.EXPERIMENTS)
TIMING = (NO_HOLD, FIXED_EARLY, LEARNED_DIRECT, FIXED_AT, FIXED_LATE)
RELEASE_TIMING = "release-timing"
TRAJECTORY = "trajectory"
SEEDS = "seeds"


def measures_of(report: dict) -> tuple[str, ...]:
    chosen = tuple(report["family"]["measures"])
    return chosen if chosen else GRAMMAR


def dot_row(axis: Axes, row: int, cell: dict) -> None:
    values = np.asarray(cell["differences"], dtype=float)
    jitter = np.linspace(-0.12, 0.12, len(values)) if len(values) > 1 else np.zeros(1)
    axis.scatter(
        values, row + jitter[: len(values)], s=10, color=MUTED, alpha=0.7, zorder=2
    )
    test = cell["test"]
    if np.isfinite(test["mean"]):
        axis.errorbar(
            test["mean"],
            row,
            xerr=[[test["mean"] - test["low"]], [test["high"] - test["mean"]]],
            color=INK,
            marker="o",
            capsize=2,
            zorder=3,
        )


def seed_panel(axis: Axes, report: dict, measure: str) -> None:
    labels = list(report["contrasts"])
    for row, name in enumerate(labels):
        dot_row(axis, row, report["contrasts"][name][measure])
    axis.set_yticks(range(len(labels)), labels)
    axis.invert_yaxis()
    zero_line(axis, True)
    axis.set_xlabel(f"{label(measure)}, paired difference")
    axis.xaxis.set_major_locator(MaxNLocator(4))


def seed_figure(inputs: Inputs, experiment: str) -> Path:
    report = inputs.report(experiment)
    measures = measures_of(report)
    rows = len(report["contrasts"])
    figure, axes = styled(1, len(measures), TEXT_WIDTH, 0.9 + 0.3 * rows)
    for k, measure in enumerate(measures):
        seed_panel(axes[0, k], report, measure)
        panel(axes[0, k], k)
        if k:
            axes[0, k].set_yticklabels([])
    figure.supxlabel(
        "+ = first arm better; dots are seeds, bar is mean and 95% CI", fontsize=7
    )
    return saved(inputs, figure, f"{SEEDS}-{experiment}")


def seeds(inputs: Inputs) -> list[Path]:
    present = [name for name in EXPERIMENTS if inputs.report_path(name).is_file()]
    return [seed_figure(inputs, name) for name in present]


def timing_points(
    inputs: Inputs, arm: str, measure: str
) -> tuple[np.ndarray, np.ndarray]:
    report = inputs.report(MixAblation.name if arm == NO_HOLD else ReleaseTiming.name)
    values = np.asarray(report["arms"][arm]["per_seed"][measure], dtype=float)
    positions = []
    for replicate in range(len(values)):
        found = inputs.training(arm, replicate)
        released = 0 if arm == NO_HOLD else found.released_at
        positions.append(np.nan if released is None else released / found.steps)
    return np.asarray(positions), values


def timing_arms(inputs: Inputs) -> tuple[str, ...]:
    if inputs.report_path(MixAblation.name).is_file():
        return TIMING
    print(
        f"{RELEASE_TIMING}: {MixAblation.name} report missing, drawing without {NO_HOLD}"
    )
    return tuple(arm for arm in TIMING if arm != NO_HOLD)


def timing_panel(inputs: Inputs, axis: Axes, measure: str) -> None:
    for arm in timing_arms(inputs):
        positions, values = timing_points(inputs, arm, measure)
        axis.scatter(
            positions,
            values,
            s=18,
            color=COLOURS[arm],
            marker=MARKERS[arm],
            label=arm,
            zorder=2,
        )
    axis.set_xlabel("release step, fraction of training")
    axis.set_ylabel(label(measure))


def release_timing(inputs: Inputs) -> list[Path]:
    figure, axes = styled(1, len(GRAMMAR), TEXT_WIDTH, 2.2)
    for k, measure in enumerate(GRAMMAR):
        timing_panel(inputs, axes[0, k], measure)
        panel(axes[0, k], k)
    axes[0, 0].legend(loc="best", fontsize=7)
    return [saved(inputs, figure, RELEASE_TIMING)]


def trajectory_points(report: dict, measure: str) -> tuple[list[float], list[dict]]:
    points = sorted((float(at), found) for at, found in report["checkpoints"].items())
    fractions = [at for at, _ in points] + [1.0]
    tests = [
        found["contrasts"][LEARNED_VS_BASELINE][measure]["test"] for _, found in points
    ]
    return fractions, tests + [
        report["contrasts"][LEARNED_VS_BASELINE][measure]["test"]
    ]


def trajectory(inputs: Inputs) -> list[Path]:
    report = inputs.report(Trajectory.name)
    figure, axes = styled(1, len(GRAMMAR), TEXT_WIDTH, 2.0)
    for k, measure in enumerate(GRAMMAR):
        axis = axes[0, k]
        fractions, tests = trajectory_points(report, measure)
        means = np.array([t["mean"] for t in tests])
        low = means - np.array([t["low"] for t in tests])
        high = np.array([t["high"] for t in tests]) - means
        axis.errorbar(
            fractions,
            means,
            yerr=[low, high],
            color=COLOURS[LEARNED_DIRECT],
            linestyle=LINESTYLES[LEARNED_DIRECT],
            marker="o",
            capsize=2,
        )
        zero_line(axis, False)
        panel(axis, k)
        axis.set_xlabel("fraction of training (learned $-$ baseline)")
        axis.set_ylabel(f"{label(measure)} gap")
        axis.set_xlim(0, 1.05)
    return [saved(inputs, figure, TRAJECTORY)]
