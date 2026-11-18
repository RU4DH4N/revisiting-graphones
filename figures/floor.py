import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from matplotlib.axes import Axes
from matplotlib.ticker import FixedFormatter, FixedLocator, NullLocator
from scipy import stats

from data.bilingual import BilingualCorpus
from data.graphones import Tables
from experiments.arm_names import LEARNED_DIRECT, LEARNED_SPLIT_DIRECT
from figures.common import (
    INK,
    MUTED,
    RULE,
    STREAM_COLOURS,
    STREAM_MARKERS,
    STREAM_NAMES,
    TEXT_WIDTH,
    Inputs,
    panel,
    saved,
    styled,
    written,
)
from figures.gates import language_rows
from figures.tables import TABLES, macro

SHARE_FLOOR = "share-floor"
NUMBERS = "share-floor-numbers"
LETTER = 0
LEVEL = 0.95
GROUPS = ((LEARNED_DIRECT, "three"), (LEARNED_SPLIT_DIRECT, "two"))
TICKS = (
    0.0002,
    0.0005,
    0.001,
    0.002,
    0.005,
    0.01,
    0.02,
    0.05,
    0.1,
    0.2,
    0.5,
    0.8,
    0.9,
)
GAP = 0.6
SPREAD = 0.28


@dataclass(frozen=True)
class Fold:
    ratio: float
    low: float
    high: float
    above: int
    seeds: int


def trained(inputs: Inputs, arm: str) -> bool:
    return all(
        inputs.layout.gates(arm, replicate).is_file()
        for replicate in range(inputs.seeds(arm))
    )


def stream_sums(
    inputs: Inputs, arm: str, rows: np.ndarray, uses: np.ndarray
) -> np.ndarray:
    found = np.stack(
        [
            uses
            @ np.square(inputs.gates(arm, replicate).final[rows].astype(np.float64))
            for replicate in range(inputs.seeds(arm))
        ]
    )
    if not (found > 0).all():
        raise ValueError(f"a stream of {arm} has no weight at all in some seed")
    return found


def shares(sums: np.ndarray) -> np.ndarray:
    return sums / sums.sum(axis=1, keepdims=True)


def logits(sums: np.ndarray, column: int) -> np.ndarray:
    return np.log(sums[:, column]) - np.log(sums.sum(axis=1) - sums[:, column])


def half_width(spread: float, freedom: float) -> float:
    return float(stats.t.ppf(0.5 + LEVEL / 2, freedom)) * spread


def paired_log_ratio(tested: np.ndarray, floor: np.ndarray) -> Fold:
    logs = np.log(tested) - np.log(floor)
    n = len(logs)
    centre = float(logs.mean())
    half = half_width(float(logs.std(ddof=1)) / math.sqrt(n), n - 1)
    return Fold(
        math.exp(centre),
        math.exp(centre - half),
        math.exp(centre + half),
        int((logs > 0).sum()),
        n,
    )


# https://doi.org/10.1093/biomet/34.1-2.28
def welch(tested: np.ndarray, floor: np.ndarray) -> Fold:
    a, b = tested.var(ddof=1) / len(tested), floor.var(ddof=1) / len(floor)
    freedom = (a + b) ** 2 / (a**2 / (len(tested) - 1) + b**2 / (len(floor) - 1))
    centre = float(tested.mean() - floor.mean())
    half = half_width(math.sqrt(a + b), freedom)
    return Fold(
        math.exp(centre),
        math.exp(centre - half),
        math.exp(centre + half),
        int((tested > floor.max()).sum()),
        len(tested),
    )


def expit(value: float) -> float:
    return 1 / (1 + math.exp(-value))


def anchored(sums: np.ndarray, column: int) -> tuple[float, float, float]:
    found = logits(sums, column)
    n = len(found)
    centre = float(found.mean())
    half = half_width(float(found.std(ddof=1)) / math.sqrt(n), n - 1)
    return expit(centre), expit(centre - half), expit(centre + half)


def share_lines(name: str, sums: np.ndarray, column: int) -> list[str]:
    share, low, high = anchored(sums, column)
    return [
        f"\\newcommand{{\\{name}Share}}{{{share:.3f}}}",
        f"\\newcommand{{\\{name}ShareLow}}{{{low:.3f}}}",
        f"\\newcommand{{\\{name}ShareHigh}}{{{high:.3f}}}",
    ]


def fold_lines(name: str, found: Fold) -> list[str]:
    return [
        f"\\newcommand{{\\{name}Ratio}}{{{found.ratio:.1f}}}",
        f"\\newcommand{{\\{name}RatioLow}}{{{found.low:.1f}}}",
        f"\\newcommand{{\\{name}RatioHigh}}{{{found.high:.1f}}}",
        f"\\newcommand{{\\{name}Above}}{{{found.above}}}",
        f"\\newcommand{{\\{name}Seeds}}{{{found.seeds}}}",
    ]


def language_macros(
    prefix: str, language: str, sums: dict[str, np.ndarray]
) -> list[str]:
    three = sums[LEARNED_DIRECT]
    lines = []
    for column, stream in enumerate(STREAM_NAMES):
        name = macro(prefix, "floor", "three", language, stream)
        lines += share_lines(name, three, column)
        if column != LETTER:
            lines += fold_lines(
                name, paired_log_ratio(three[:, column], three[:, LETTER])
            )
    if LEARNED_SPLIT_DIRECT in sums:
        two = sums[LEARNED_SPLIT_DIRECT]
        for column in range(two.shape[1]):
            name = macro(prefix, "floor", "two", language, STREAM_NAMES[column])
            lines += share_lines(name, two, column)
        name = macro(prefix, "floor", "two", language, STREAM_NAMES[LETTER])
        lines += fold_lines(name, welch(logits(two, LETTER), logits(three, LETTER)))
    return lines


def column(axis: Axes, at: float, values: np.ndarray, stream: int) -> None:
    offsets = np.linspace(-SPREAD, SPREAD, len(values))
    axis.scatter(
        at + offsets,
        values,
        s=14,
        color=STREAM_COLOURS[stream],
        marker=STREAM_MARKERS[stream],
        linewidths=0,
        zorder=3,
    )
    centre = math.exp(float(np.log(values).mean()))
    axis.plot(
        [at - 0.4, at + 0.4], [centre, centre], color=INK, linewidth=1.0, zorder=4
    )


def group(axis: Axes, start: float, found: np.ndarray, first: bool) -> list[float]:
    streams = found.shape[1]
    places = [start + k for k in range(streams)]
    for k, at in enumerate(places):
        column(axis, at, found[:, k], k)
    axis.plot(
        [places[0] - 0.5, places[-1] + 0.5],
        [1 / streams, 1 / streams],
        color=MUTED,
        linewidth=0.8,
        linestyle="--",
        zorder=2,
        label="equal start" if first else None,
    )
    return places


def share_panel(axis: Axes, sums: dict[str, np.ndarray]) -> tuple[float, float]:
    three = shares(sums[LEARNED_DIRECT])
    floor = three[:, LETTER]
    axis.axhspan(
        floor.min(),
        floor.max(),
        color=RULE,
        zorder=1,
        label="floor: letter code beside the joint code",
    )
    places = group(axis, 0.0, three, True)
    names = [*STREAM_NAMES]
    drawn = [three]
    if LEARNED_SPLIT_DIRECT in sums:
        two = shares(sums[LEARNED_SPLIT_DIRECT])
        places += group(axis, len(STREAM_NAMES) + GAP, two, False)
        names += [*STREAM_NAMES[: two.shape[1]]]
        drawn.append(two)
    low = min(float(found.min()) for found in drawn)
    high = max(float(found.max()) for found in drawn)
    axis.set_yscale("logit")
    axis.yaxis.set_major_locator(FixedLocator(TICKS))
    axis.yaxis.set_major_formatter(FixedFormatter([f"{tick:g}" for tick in TICKS]))
    axis.yaxis.set_minor_locator(NullLocator())
    axis.set_xticks(places, names)
    axis.set_xlim(places[0] - 0.7, places[-1] + 0.7)
    axis.grid(axis="x", visible=False)
    return low, high


def language_sums(
    inputs: Inputs, tables: Tables, k: int, arms: list[str]
) -> dict[str, np.ndarray]:
    rows = language_rows(tables, k)
    uses = tables.uses[rows].astype(np.float64)
    return {arm: stream_sums(inputs, arm, rows, uses) for arm in arms}


def share_floor(inputs: Inputs) -> list[Path]:
    arms = [arm for arm, _ in GROUPS if trained(inputs, arm)]
    if LEARNED_DIRECT not in arms:
        print(f"{SHARE_FLOOR}: no finished {LEARNED_DIRECT} runs, nothing drawn")
        return []
    tables = inputs.tables()
    languages = list(BilingualCorpus(inputs.layout.corpus).languages)
    width = TEXT_WIDTH if len(languages) > 1 else 0.6 * TEXT_WIDTH
    figure, axes = styled(1, len(languages), width, 2.5)
    lines = []
    ranges = []
    for k, language in enumerate(languages):
        sums = language_sums(inputs, tables, k, arms)
        ranges.append(share_panel(axes[0, k], sums))
        axes[0, k].set_title(language, fontsize=8, color=INK)
        if len(languages) > 1:
            panel(axes[0, k], k)
        lines += language_macros(inputs.config_path.stem, language, sums)
    low = min(found[0] for found in ranges)
    high = max(found[1] for found in ranges)
    for axis in axes[0]:
        axis.set_ylim(low * 0.6, 1 - (1 - high) * 0.6)
    axes[0, 0].set_ylabel("share of gate weight")
    handles, names = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, names, loc="outside upper center", ncols=len(names))
    codes = "three codes, then two" if len(arms) > 1 else "three codes"
    figure.supxlabel(f"{codes}; dots are seeds, bar is the geometric mean", fontsize=7)
    return [
        saved(inputs, figure, SHARE_FLOOR),
        written(inputs.folder / TABLES / f"{NUMBERS}.tex", "\n".join(lines) + "\n"),
    ]
