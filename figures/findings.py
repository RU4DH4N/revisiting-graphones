from dataclasses import dataclass
from pathlib import Path

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import FixedFormatter, FixedLocator, MaxNLocator, NullLocator

from experiments.arm_names import LEARNED_DIRECT
from experiments.graphones_vs_letters import GraphonesVsLetters
from experiments.mix_ablation import MixAblation
from experiments.one_language import OneLanguage
from figures.common import INK, MUTED, RULE, Inputs, saved, styled
from figures.floor import LETTER, paired_log_ratio, shares, stream_sums, trained
from figures.gates import language_rows
from figures.tables import number, probability
from framework.paths import ARTEFACTS
from workers.layout import Layout
from workers.records import (
    BITS_PER_BYTE,
    CKA,
    MINIMAL_PAIRS,
    POS_SELECTIVITY,
    POS_TRANSFER,
)
from workers.settings import read_config

BENCHMARK = "finding-benchmark"
SHARES = "finding-shares"
PAIRS = "finding-pairs"
NEEDS = (OneLanguage.name,)
PAGE_WIDTH = 6.5
COLUMN_WIDTH = 3.1
SEED = "#8a8a8a"
FAINT = "#c9c9c9"
ESTIMATE = "#0072B2"
STREAM_TONES = ("#E69F00", "#0072B2", "#111111")
STREAM_SHAPES = ("o", "s", "^")
ROW_HEIGHT = 0.3
BENCHMARK_ROWS = (("graphones_vs_bytes", "bytes"), ("graphones_vs_letters", "letters"))
BENCHMARK_PANELS = (
    (BITS_PER_BYTE, "reduction in bits per byte"),
    (MINIMAL_PAIRS, "gain in grammatical judgement"),
    (POS_SELECTIVITY, "gain in POS selectivity"),
)
SHARED = (("es-pt", "es-pt, many"), ("en-fr", "en-fr, fewer"), ("en-ru", "en-ru, none"))
STREAMS_VS_LETTERS = "split_vs_letters"
WITHOUT_JOINT = "learned-split-direct_vs_full"
PAIR_PANELS = (
    (
        GraphonesVsLetters.name,
        STREAMS_VS_LETTERS,
        BITS_PER_BYTE,
        "reduction in bits per byte",
    ),
    (GraphonesVsLetters.name, STREAMS_VS_LETTERS, POS_TRANSFER, "gain in POS transfer"),
    (GraphonesVsLetters.name, STREAMS_VS_LETTERS, CKA, "gain in CKA"),
)
STREAM_LABELS = ("letter\n(unused)", "sound", "joint")
TICKS = (0.0002, 0.001, 0.005, 0.02, 0.1, 0.33, 0.6, 0.8)


@dataclass(frozen=True)
class Row:
    label: str
    seeds: np.ndarray
    mean: float
    low: float
    high: float
    note: str


def contrast_row(label: str, cell: dict, corrected: float | None) -> Row:
    test = cell["test"]
    interval = f"[${number(test['low'])}$, ${number(test['high'])}$]"
    tested = "" if corrected is None else f"   $p$ = {probability(corrected)}"
    note = (
        f"${number(test['mean'])}$  {interval}{tested}   {cell['wins']} of {test['n']}"
    )
    return Row(
        label,
        np.asarray(cell["differences"], dtype=float),
        test["mean"],
        test["low"],
        test["high"],
        note,
    )


def forest(axis: Axes, rows: list[Row], title: str) -> None:
    for at, row in enumerate(rows):
        jitter = np.linspace(-0.16, 0.16, len(row.seeds))
        axis.scatter(
            row.seeds,
            at + jitter,
            s=13,
            facecolors="none",
            edgecolors=SEED,
            linewidths=0.7,
            zorder=2,
        )
        axis.errorbar(
            row.mean,
            at,
            xerr=[[row.mean - row.low], [row.high - row.mean]],
            color=ESTIMATE,
            marker="o",
            markersize=4.5,
            linewidth=1.3,
            capsize=2.5,
            zorder=3,
        )
        axis.annotate(
            row.note,
            xy=(1.03, at),
            xycoords=("axes fraction", "data"),
            va="center",
            ha="left",
            fontsize=7,
            color=INK,
            annotation_clip=False,
        )
    axis.axvline(0.0, color=MUTED, linewidth=0.8, linestyle="--", zorder=1)
    axis.set_yticks(range(len(rows)), [row.label for row in rows])
    axis.set_ylim(len(rows) - 0.5, -0.5)
    axis.tick_params(axis="y", length=0, labelsize=8, labelcolor=INK)
    axis.grid(axis="y", visible=False)
    axis.xaxis.set_major_locator(MaxNLocator(4))
    axis.set_title(title, fontsize=8, color=INK, loc="left")
    axis.spines["left"].set_visible(False)


def stacked(panels: int, rows: int) -> tuple[Figure, np.ndarray]:
    height = panels * (0.5 + ROW_HEIGHT * rows)
    figure, axes = styled(panels, 1, PAGE_WIDTH, height)
    figure.set_layout_engine(None)
    figure.subplots_adjust(
        left=0.15,
        right=0.66,
        top=1 - 0.22 / height,
        bottom=0.3 / height,
        hspace=0.5 / (ROW_HEIGHT * rows),
    )
    return figure, axes[:, 0]


def benchmark(inputs: Inputs) -> list[Path]:
    report = inputs.report(OneLanguage.name)
    figure, axes = stacked(len(BENCHMARK_PANELS), len(BENCHMARK_ROWS))
    for axis, (measure, title) in zip(axes, BENCHMARK_PANELS, strict=True):
        rows = [
            contrast_row(
                f"against {label}",
                report["contrasts"][contrast][measure],
                report["family"]["holm"][f"{contrast}/{measure}"],
            )
            for contrast, label in BENCHMARK_ROWS
        ]
        forest(axis, rows, title)
    return [saved(inputs, figure, BENCHMARK)]


def pair_inputs(inputs: Inputs, pair: str) -> Inputs:
    path = inputs.config_path.parent / f"{pair}.json"
    return Inputs(read_config(path), path, Layout(ARTEFACTS / pair), inputs.split)


def pairs(inputs: Inputs) -> list[Path]:
    found = {pair: pair_inputs(inputs, pair) for pair, _ in SHARED}
    lacking = [
        pair
        for pair, source in found.items()
        if not source.report_path(GraphonesVsLetters.name).is_file()
        or not source.report_path(MixAblation.name).is_file()
    ]
    if lacking:
        print(f"{PAIRS}: no finished reports for {', '.join(lacking)}, nothing drawn")
        return []
    panels = [
        *PAIR_PANELS,
        (MixAblation.name, WITHOUT_JOINT, BITS_PER_BYTE, "reduction in bits per byte"),
    ]
    figure, axes = stacked(len(panels), len(SHARED))
    for axis, (experiment, contrast, measure, title) in zip(axes, panels, strict=True):
        rows = [
            contrast_row(
                label,
                found[pair].report(experiment)["contrasts"][contrast][measure],
                None,
            )
            for pair, label in SHARED
        ]
        forest(axis, rows, title)
    axes[0].set_title(
        "parallel streams against letters: reduction in bits per byte",
        fontsize=8,
        color=INK,
        loc="left",
    )
    axes[-1].set_title(
        "gate without the joint stream against with it: reduction in bits per byte",
        fontsize=8,
        color=INK,
        loc="left",
    )
    return [saved(inputs, figure, PAIRS)]


def share_points(axis: Axes, found: np.ndarray) -> None:
    places = np.arange(found.shape[1])
    for seed in found:
        axis.plot(places, seed, color=FAINT, linewidth=0.6, zorder=2)
    for at in places:
        axis.scatter(
            np.full(len(found), at),
            found[:, at],
            s=16,
            marker=STREAM_SHAPES[at],
            facecolors="white",
            edgecolors=STREAM_TONES[at],
            linewidths=0.9,
            zorder=3,
        )
        centre = float(np.exp(np.log(found[:, at]).mean()))
        axis.plot(
            [at - 0.28, at + 0.28], [centre, centre], color=INK, linewidth=1.6, zorder=4
        )


def share_labels(axis: Axes, sums: np.ndarray, found: np.ndarray) -> None:
    for at in range(1, found.shape[1]):
        ratio = paired_log_ratio(sums[:, at], sums[:, LETTER])
        axis.annotate(
            f"{ratio.ratio:.0f}$\\times$ unused\n[{ratio.low:.0f}, {ratio.high:.0f}]",
            xy=(at, found[:, at].max()),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=6.5,
            color=INK,
        )
    axis.annotate(
        "equal start",
        xy=(found.shape[1] - 0.55, 1 / found.shape[1]),
        xytext=(0, -2),
        textcoords="offset points",
        ha="right",
        va="top",
        fontsize=6.5,
        color=MUTED,
    )


def share_figure(inputs: Inputs) -> list[Path]:
    if not trained(inputs, LEARNED_DIRECT):
        print(f"{SHARES}: no finished {LEARNED_DIRECT} runs, nothing drawn")
        return []
    tables = inputs.tables()
    rows = language_rows(tables, 0)
    uses = tables.uses[rows].astype(np.float64)
    sums = stream_sums(inputs, LEARNED_DIRECT, rows, uses)
    found = shares(sums)
    figure, axes = styled(1, 1, COLUMN_WIDTH, 2.7)
    axis = axes[0, 0]
    floor = found[:, LETTER]
    axis.axhspan(floor.min(), floor.max(), color=RULE, alpha=0.6, zorder=1)
    axis.axhline(
        1 / found.shape[1], color=MUTED, linewidth=0.8, linestyle="--", zorder=1
    )
    share_points(axis, found)
    share_labels(axis, sums, found)
    axis.set_yscale("logit")
    axis.set_ylim(float(found.min()) * 0.6, 0.93)
    axis.yaxis.set_major_locator(FixedLocator(TICKS))
    axis.yaxis.set_major_formatter(FixedFormatter([f"{tick:g}" for tick in TICKS]))
    axis.yaxis.set_minor_locator(NullLocator())
    axis.set_xticks(range(found.shape[1]), STREAM_LABELS[: found.shape[1]])
    axis.set_xlim(-0.6, found.shape[1] - 0.4)
    axis.tick_params(axis="x", length=0, labelsize=8, labelcolor=INK)
    axis.grid(axis="x", visible=False)
    axis.set_ylabel("share of the gate's weight")
    return [saved(inputs, figure, SHARES)]
