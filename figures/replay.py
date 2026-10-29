from pathlib import Path

import numpy as np
from matplotlib.axes import Axes

from experiments.arm_names import BASELINE
from experiments.headline import Headline
from figures.common import INK, TEXT_WIDTH, Inputs, panel, saved, styled
from workers.records import GaugePoint
from workers.train import Hold

NEEDS = (Headline.name,)
RELEASE_REPLAY = "release-replay"
FALLS = (0.05, 0.075, 0.1, 0.15, 0.2)
EVERY = (1, 2, 4)
# https://doi.org/10.1371/journal.pone.0199239
RAMP = "cividis"


def fired(points: list[GaugePoint], fall: float, every: int, noisy: bool) -> int | None:
    hold = Hold(True, fall, True, None)
    for point in [points[0], *points[1:][every - 1 :: every]]:
        if hold.observe(point.language, point.noise if noisy else 0.0, point.step):
            return point.step
    return None


def grid(inputs: Inputs, noisy: bool) -> tuple[np.ndarray, np.ndarray]:
    runs = [inputs.training(BASELINE, r) for r in range(inputs.seeds(BASELINE))]
    median = np.full((len(FALLS), len(EVERY)), np.nan)
    count = np.zeros((len(FALLS), len(EVERY)), dtype=int)
    for i, fall in enumerate(FALLS):
        for j, every in enumerate(EVERY):
            steps = [fired(run.gauge, fall, every, noisy) for run in runs]
            fractions = [
                s / run.steps
                for s, run in zip(steps, runs, strict=True)
                if s is not None
            ]
            count[i, j] = len(fractions)
            median[i, j] = np.median(fractions) if fractions else np.nan
    return median, count


def annotate(
    axis: Axes, median: np.ndarray, count: np.ndarray, seeds: int, top: float
) -> None:
    for i in range(len(FALLS)):
        for j in range(len(EVERY)):
            value = median[i, j]
            text = "never" if np.isnan(value) else f"{value:.3f}"
            ink = "white" if np.isfinite(value) and value < top / 2 else INK
            axis.text(
                j,
                i,
                f"{text}\n{count[i, j]}/{seeds}",
                ha="center",
                va="center",
                fontsize=6,
                color=ink,
            )


def release_replay(inputs: Inputs) -> list[Path]:
    figure, axes = styled(1, 2, TEXT_WIDTH, 2.3)
    grids = [grid(inputs, noisy) for noisy in (True, False)]
    finite = [m[np.isfinite(m)] for m, _ in grids]
    top = max([0.2, *(float(v.max()) for v in finite if len(v))])
    interval = inputs.config.gauge.interval
    images = []
    for k, (median, count) in enumerate(grids):
        axis = axes[0, k]
        image = axis.imshow(median, cmap=RAMP, vmin=0, vmax=top, aspect="auto")
        annotate(axis, median, count, inputs.seeds(BASELINE), top)
        axis.set_xticks(range(len(EVERY)), [str(e * interval) for e in EVERY])
        axis.set_yticks(range(len(FALLS)), [str(f) for f in FALLS])
        axis.set_xlabel("steps between readings")
        axis.set_ylabel("fall")
        panel(axis, k)
        axis.grid(False)
        images.append(image)
    figure.colorbar(
        images[-1], ax=axes[0, :], label="median release, fraction of training"
    )
    return [saved(inputs, figure, RELEASE_REPLAY)]
