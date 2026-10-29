from pathlib import Path

import numpy as np
from matplotlib.axes import Axes

from experiments.arm_names import BASELINE, LEARNED_DIRECT
from experiments.headline import Headline
from figures.common import (
    COLOURS,
    LINESTYLES,
    MUTED,
    RULE,
    STREAM_COLOURS,
    STREAM_LINESTYLES,
    STREAM_NAMES,
    TEXT_WIDTH,
    Inputs,
    saved,
    styled,
)

NEEDS = (Headline.name,)
LANGUAGE_SHARE = "language-share"
GATE_SHARES = "gate-shares"


def mark_checkpoints(inputs: Inputs, axis: Axes) -> None:
    for fraction in inputs.config.checkpoints.fractions:
        axis.axvline(fraction, color=RULE, linewidth=0.8, linestyle="--")


def mark_release(axis: Axes, fraction: float | None, colour: str) -> None:
    if fraction is not None:
        axis.axvline(fraction, color=colour, linewidth=0.6, alpha=0.6)


def share_lines(inputs: Inputs, axis: Axes, arm: str) -> None:
    for replicate in range(inputs.seeds(arm)):
        found = inputs.training(arm, replicate)
        steps = np.array([point.step for point in found.gauge]) / found.steps
        share = np.array([point.language for point in found.gauge])
        axis.plot(
            steps,
            share,
            color=COLOURS[arm],
            linestyle=LINESTYLES[arm],
            linewidth=0.9,
            alpha=0.7,
            label=arm if replicate == 0 else None,
        )
        released = found.released_at
        mark_release(
            axis, None if released is None else released / found.steps, COLOURS[arm]
        )


def language_share(inputs: Inputs) -> list[Path]:
    figure, axes = styled(1, 1, 0.6 * TEXT_WIDTH, 2.0)
    axis = axes[0, 0]
    for arm in (BASELINE, LEARNED_DIRECT):
        share_lines(inputs, axis, arm)
    mark_checkpoints(inputs, axis)
    axis.set_xlabel("fraction of training")
    axis.set_ylabel("language share")
    axis.set_xlim(0, 1)
    axis.legend(loc="upper right")
    return [saved(inputs, figure, LANGUAGE_SHARE)]


def usage_shares(
    directions: np.ndarray, rows: np.ndarray, uses: np.ndarray
) -> np.ndarray:
    return np.average(np.square(directions[rows]), axis=0, weights=uses[rows])


def seed_shares(
    inputs: Inputs, replicate: int, rows: np.ndarray, uses: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    gates = inputs.gates(LEARNED_DIRECT, replicate)
    shares = np.stack([usage_shares(table, rows, uses) for table in gates.snapshots])
    return gates.snapshot_steps / inputs.training(
        LEARNED_DIRECT, replicate
    ).steps, shares


def gate_shares(inputs: Inputs) -> list[Path]:
    tables = inputs.tables()
    rows = np.flatnonzero(tables.has & (tables.uses > 0))
    figure, axes = styled(1, 1, 0.6 * TEXT_WIDTH, 2.0)
    axis = axes[0, 0]
    for replicate in range(inputs.seeds(LEARNED_DIRECT)):
        steps, shares = seed_shares(inputs, replicate, rows, tables.uses)
        for k, colour in enumerate(STREAM_COLOURS):
            name = STREAM_NAMES[k] if replicate == 0 else None
            axis.plot(
                steps,
                shares[:, k],
                color=colour,
                linestyle=STREAM_LINESTYLES[k],
                linewidth=0.9,
                alpha=0.8,
                label=name,
            )
        released = inputs.training(LEARNED_DIRECT, replicate).released_at
        steps_total = inputs.training(LEARNED_DIRECT, replicate).steps
        mark_release(axis, None if released is None else released / steps_total, MUTED)
    mark_checkpoints(inputs, axis)
    axis.set_xlabel("fraction of training")
    axis.set_ylabel("stream share")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.legend(loc="center right")
    return [saved(inputs, figure, GATE_SHARES)]
