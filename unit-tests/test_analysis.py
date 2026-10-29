import json
import math
from pathlib import Path

import pytest
from conftest import arm, raw_config

from models.graphone import EQUAL, SPLIT
from workers.analysis import (
    LANGUAGE_LATE,
    ArmResults,
    Summary,
    contrasts,
    summary,
    window,
)
from workers.layout import Layout
from workers.records import SCORES, GaugePoint
from workers.settings import parse

GAUGE = [GaugePoint(step, step / 100, 0.0) for step in range(10, 101, 10)]


def test_a_summary_reports_the_sample_spread_of_the_finite_seeds() -> None:
    assert summary([1.0, 2.0, 3.0, math.nan]) == Summary(2.0, 1.0, 3)


def test_a_window_averages_only_the_gauge_readings_inside_it() -> None:
    assert window(GAUGE, 0.5, 1.0, 100) == pytest.approx(0.8)
    assert window(GAUGE, 0.1, 0.45, 100) == pytest.approx(0.3)
    assert math.isnan(window(GAUGE, 0.0, 0.05, 100))


def write_seed(
    layout: Layout, name: str, replicate: int, value: float, language: float
) -> None:
    run = layout.run(name, replicate)
    (run / "selection").mkdir(parents=True)
    gauge = [{"step": 20, "language": 0.0, "noise": 0.0}] + [
        {"step": step, "language": language, "noise": 0.0} for step in (60, 100)
    ]
    metrics = {
        "arm": name,
        "replicate": replicate,
        "steps": 100,
        "seeds_together": 2,
        "seconds": 1.0,
        "device": "cpu",
        "receiver_parameters": 1,
        "released_at": None,
        "curve": [],
        "gauge": gauge,
    }
    layout.metrics(name, replicate).write_text(json.dumps(metrics))
    layout.evaluation(name, replicate, "selection").write_text(
        json.dumps(
            {
                **{measure: value for measure in SCORES},
                "by_language": {},
                "words": 1,
            }
        )
    )


def test_contrasts_compare_arms_seed_by_seed(tmp_path: Path) -> None:
    layout = Layout(tmp_path)
    paired = arm(32, SPLIT, EQUAL, seeds=2)
    config = parse(raw_config({"left": paired, "right": paired}))
    for replicate, (value, language) in enumerate(((1.0, 0.5), (1.2, 0.7))):
        write_seed(layout, "left", replicate, value, language)
        write_seed(layout, "right", replicate, value + 0.1, language - 0.2)
    arms = {
        name: ArmResults(layout, config, name, "selection", None)
        for name in ("left", "right")
    }
    found = contrasts(arms, {"gap": ("left", "right")})["gap"]
    for measure in SCORES:
        sign = -1.0 if measure == "bits_per_byte" else 1.0
        assert found[measure].test.mean == pytest.approx(-0.1 * sign)
    assert found[LANGUAGE_LATE].test.mean == pytest.approx(0.2)
