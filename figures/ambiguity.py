from pathlib import Path

import numpy as np

from experiments.arm_names import LEARNED_DIRECT
from experiments.headline import Headline
from figures.common import (
    COLOURS,
    STREAM_COLOURS,
    STREAM_NAMES,
    TEXT_WIDTH,
    Inputs,
    panel,
    saved,
    styled,
)

NEEDS = (Headline.name,)
GATE_AMBIGUITY = "gate-ambiguity"
BINS = 8
AMBIGUITY = 0


def mean_directions(inputs: Inputs) -> np.ndarray:
    finals = [
        inputs.gates(LEARNED_DIRECT, r).final
        for r in range(inputs.seeds(LEARNED_DIRECT))
    ]
    return np.mean(np.stack(finals), axis=0)


def ambiguity_bins(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ordered = weights[order]
    cumulative = (np.cumsum(ordered) - ordered / 2) / ordered.sum()
    bins = np.empty(len(values), dtype=int)
    bins[order] = np.minimum((cumulative * BINS).astype(int), BINS - 1)
    return bins


def binned(values: np.ndarray, bins: np.ndarray, weights: np.ndarray) -> np.ndarray:
    total = np.bincount(bins, weights=weights, minlength=BINS)
    return np.bincount(bins, weights=values * weights, minlength=BINS) / np.maximum(
        total, 1e-12
    )


def gate_ambiguity(inputs: Inputs) -> list[Path]:
    tables = inputs.tables()
    rows = np.flatnonzero(tables.has & (tables.uses > 0))
    uses = tables.uses[rows].astype(float)
    ambiguity = tables.features[rows, AMBIGUITY]
    bins = ambiguity_bins(ambiguity, uses)
    centres = binned(ambiguity, bins, uses)
    finals = [
        inputs.gates(LEARNED_DIRECT, r).final[rows]
        for r in range(inputs.seeds(LEARNED_DIRECT))
    ]
    figure, axes = styled(1, 2, TEXT_WIDTH, 2.0)
    for k, colour in enumerate(STREAM_COLOURS):
        shares = np.stack([binned(np.square(f[:, k]), bins, uses) for f in finals])
        axes[0, 0].plot(
            centres,
            shares.mean(axis=0),
            color=colour,
            marker="o",
            markersize=4,
            label=STREAM_NAMES[k],
        )
    negative = np.stack(
        [binned((f < 0).any(axis=1).astype(float), bins, uses) for f in finals]
    )
    axes[0, 1].plot(
        centres, negative.mean(axis=0), color=COLOURS[LEARNED_DIRECT], marker="o"
    )
    axes[0, 0].set_ylabel("stream share")
    axes[0, 0].legend(loc="best")
    axes[0, 1].set_ylabel("tokens with a negative weight")
    for k, axis in enumerate(axes[0]):
        axis.set_xlabel(r"$H(\mathrm{sound} \mid \mathrm{letter})$, bits")
        panel(axis, k)
    return [saved(inputs, figure, GATE_AMBIGUITY)]
