import math
from dataclasses import asdict, dataclass

import numpy as np

from framework.storage import read_json
from workers.layout import Layout
from workers.records import SCORES, GaugePoint, Metrics, read_evaluation, read_metrics
from workers.settings import Config
from workers.statistics import Cell, equivalence, holm, paired

LANGUAGE_LATE = "language_late"
MEASURES = (*SCORES, LANGUAGE_LATE)


def window(points: list[GaugePoint], low: float, high: float, steps: int) -> float:
    values = [p.language for p in points if low * steps < p.step <= high * steps]
    return float(np.mean(values)) if values else math.nan


@dataclass(frozen=True)
class ArmResults:
    layout: Layout
    config: Config
    name: str
    split: str
    at: float | None

    @property
    def seeds(self) -> int:
        return self.config.arms[self.name].seeds

    def training(self, replicate: int) -> Metrics:
        return read_metrics(read_json(self.layout.metrics(self.name, replicate)))

    def evaluation(self, replicate: int) -> dict:
        path = (
            self.layout.evaluation(self.name, replicate, self.split)
            if self.at is None
            else self.layout.evaluation_at(self.name, replicate, self.split, self.at)
        )
        return asdict(read_evaluation(read_json(path)))

    def measures(self, replicate: int) -> dict[str, float]:
        training = self.training(replicate)
        evaluation = self.evaluation(replicate)
        point = 1.0 if self.at is None else self.at
        late = window(
            training.gauge, self.config.analysis.late * point, point, training.steps
        )
        return {**{name: evaluation[name] for name in SCORES}, LANGUAGE_LATE: late}

    def table(self) -> dict[str, list[float]]:
        rows = [self.measures(replicate) for replicate in range(self.seeds)]
        return {name: [row[name] for row in rows] for name in MEASURES}

    def released(self) -> list[int | None]:
        return [self.training(replicate).released_at for replicate in range(self.seeds)]


@dataclass(frozen=True)
class Summary:
    mean: float
    sd: float
    seeds: int


def summary(values: list[float]) -> Summary:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    return Summary(
        float(array.mean()) if len(array) else math.nan,
        float(array.std(ddof=1)) if len(array) > 1 else math.nan,
        len(array),
    )


def arm_report(arm: ArmResults) -> dict:
    table = arm.table()
    return {
        "measures": {name: asdict(summary(values)) for name, values in table.items()},
        "per_seed": table,
    }


def cell_report(cell: Cell, tested: bool) -> dict:
    found = asdict(cell)
    if tested:
        found["equivalence"] = asdict(equivalence(cell))
    return found


def contrasts(
    arms: dict[str, ArmResults], pairs: dict[str, tuple[str, str]]
) -> dict[str, dict[str, Cell]]:
    tables = {name: arm.table() for name, arm in arms.items()}
    return {
        label: {
            name: paired(name, tables[left][name], tables[right][name])
            for name in MEASURES
        }
        for label, (left, right) in pairs.items()
    }


def family_p(cell: Cell, equivalent: bool) -> float:
    return equivalence(cell).p if equivalent else cell.test.p


def family(
    found: dict[str, dict[str, Cell]], primary: tuple[str, ...], equivalent: bool
) -> dict:
    tested = {
        f"{label}/{measure}": family_p(cells[measure], equivalent)
        for label, cells in found.items()
        for measure in primary
    }
    return {
        "measures": list(primary),
        "test": "tost" if equivalent else "paired-t",
        "holm": holm(tested),
    }


def contrast_report(
    found: dict[str, dict[str, Cell]], primary: tuple[str, ...], equivalent: bool
) -> dict:
    return {
        label: {
            measure: cell_report(cell, equivalent and measure in primary)
            for measure, cell in cells.items()
        }
        for label, cells in found.items()
    }
