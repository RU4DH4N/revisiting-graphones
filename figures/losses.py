import math
from pathlib import Path

import numpy as np
from scipy import stats

from data.bilingual import BilingualCorpus
from data.graphones import type_counts
from experiments.arm_names import BASELINE, LEARNED_DIRECT
from experiments.headline import Headline
from figures.common import (
    COLOURS,
    INK,
    MUTED,
    TEXT_WIDTH,
    Inputs,
    saved,
    smoothed,
    styled,
    zero_line,
)

NEEDS = (Headline.name,)
LOSS_GAP = "loss-gap"
LOSS_BY_FREQUENCY = "loss-by-frequency"
BANDS = 8


def paired_seeds(inputs: Inputs) -> int:
    return min(inputs.seeds(LEARNED_DIRECT), inputs.seeds(BASELINE))


def loss_gap(inputs: Inputs) -> list[Path]:
    figure, axes = styled(1, 1, 0.6 * TEXT_WIDTH, 2.0)
    axis = axes[0, 0]
    gaps = []
    for replicate in range(paired_seeds(inputs)):
        gap = inputs.steps(LEARNED_DIRECT, replicate) - inputs.steps(
            BASELINE, replicate
        )
        width = max(1, len(gap) // 100)
        line = smoothed(gap, width)
        gaps.append(line)
        steps = (np.arange(len(line)) + len(gap) - len(line)) / len(gap)
        axis.plot(steps, line, color=COLOURS[LEARNED_DIRECT], linewidth=0.6, alpha=0.35)
    mean = np.mean(np.stack(gaps), axis=0)
    total = len(inputs.steps(LEARNED_DIRECT, 0))
    steps = (np.arange(len(mean)) + total - len(mean)) / total
    axis.plot(steps, mean, color=INK, linewidth=1.4, label="mean over paired seeds")
    zero_line(axis, False)
    axis.set_xlabel("fraction of training")
    axis.set_ylabel("loss gap")
    axis.set_xlim(0, 1)
    axis.legend(loc="upper right")
    return [saved(inputs, figure, LOSS_GAP)]


def frequency_bands(
    counts: np.ndarray, languages: np.ndarray, types: np.ndarray
) -> np.ndarray:
    seen = counts[languages, types].astype(float)
    edges = np.quantile(np.log1p(seen), np.linspace(0, 1, BANDS + 1))
    return np.clip(
        np.searchsorted(edges, np.log1p(seen), side="right") - 1, 0, BANDS - 1
    )


def band_rates(scores: dict[str, np.ndarray], bands: np.ndarray) -> np.ndarray:
    nats = np.bincount(bands, weights=scores["nats"], minlength=BANDS)
    symbols = np.bincount(bands, weights=scores["symbols"], minlength=BANDS)
    return nats / np.maximum(symbols, 1) / math.log(2)


def band_gaps(inputs: Inputs) -> np.ndarray:
    counts = type_counts(BilingualCorpus(inputs.layout.corpus))
    gaps = []
    for replicate in range(paired_seeds(inputs)):
        learned = inputs.scores(LEARNED_DIRECT, replicate)
        baseline = inputs.scores(BASELINE, replicate)
        bands = frequency_bands(counts, learned["languages"], learned["types"])
        gaps.append(band_rates(learned, bands) - band_rates(baseline, bands))
    return np.stack(gaps)


def loss_by_frequency(inputs: Inputs) -> list[Path]:
    gaps = band_gaps(inputs)
    mean = gaps.mean(axis=0)
    error = (
        gaps.std(axis=0, ddof=1) / math.sqrt(len(gaps))
        if len(gaps) > 1
        else np.zeros(BANDS)
    )
    reach = float(stats.t.ppf(0.975, max(len(gaps) - 1, 1))) * error
    figure, axes = styled(1, 1, 0.6 * TEXT_WIDTH, 2.0)
    axis = axes[0, 0]
    bands = np.arange(1, BANDS + 1)
    for gap in gaps:
        axis.plot(bands, gap, color=MUTED, linewidth=0.5, alpha=0.4)
    axis.errorbar(
        bands,
        mean,
        yerr=reach,
        color=COLOURS[LEARNED_DIRECT],
        marker="o",
        markersize=4,
        capsize=2,
    )
    zero_line(axis, False)
    axis.set_xticks(bands)
    axis.set_xlabel("training-frequency band of the word (1 = rarest)")
    axis.set_ylabel("bits per symbol gap")
    return [saved(inputs, figure, LOSS_BY_FREQUENCY)]
