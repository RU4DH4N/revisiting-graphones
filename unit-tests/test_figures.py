import json
import shutil
from pathlib import Path

import numpy as np
from conftest import Tiny

from data.graphones import read_tables
from figures.ambiguity import BINS, ambiguity_bins, binned
from figures.common import Inputs, smoothed
from figures.compute import cost, counted_batch
from figures.losses import BANDS, frequency_bands
from figures.replay import fired
from figures.tables import macro, probability
from workers.design import sizing_record
from workers.layout import Layout
from workers.records import GaugePoint
from workers.settings import parse


def readings(shares: list[float]) -> list[GaugePoint]:
    return [GaugePoint(1 + 50 * k, share, 0.0) for k, share in enumerate(shares)]


def test_the_replay_fires_where_the_training_rule_would() -> None:
    points = readings([0.1, 0.5, 0.9, 0.85, 0.7])
    assert fired(points, 0.05, 1, True) == 151
    assert fired(points, 0.1, 1, True) == 201
    assert fired(points, 0.3, 1, True) is None
    assert fired(points, 0.05, 2, True) == 201


def test_a_share_that_never_falls_never_fires() -> None:
    assert fired(readings([0.1, 0.3, 0.6, 0.9]), 0.1, 1, True) is None


def test_ambiguity_bins_hold_equal_usage() -> None:
    values = np.linspace(0, 3, 800)
    weights = np.ones(800)
    bins = ambiguity_bins(values, weights)
    assert np.bincount(bins, minlength=BINS).tolist() == [100] * BINS
    assert binned(values, bins, weights)[0] < binned(values, bins, weights)[-1]


def test_frequency_bands_run_from_rare_to_common() -> None:
    counts = np.array([[1, 10, 100, 1000, 10000, 5, 50, 500]])
    types = np.arange(8)
    bands = frequency_bands(counts, np.zeros(8, dtype=int), types)
    assert bands.min() == 0 and bands.max() == BANDS - 1
    assert bands[0] < bands[4]


def test_smoothing_keeps_a_constant_and_shortens_by_the_window() -> None:
    assert np.allclose(smoothed(np.full(10, 2.0), 4), 2.0)
    assert len(smoothed(np.arange(10.0), 4)) == 7


def test_macros_are_letters_only_and_probabilities_have_a_floor() -> None:
    assert macro("learned-mix", "learned_vs_baseline", "minimal_pairs") == (
        "LearnedMixLearnedVsBaselineMinimalPairs"
    )
    assert probability(0.0001) == "<0.001"
    assert probability(float("nan")) == "--"
    assert probability(0.0123) == "0.012"


def test_the_compute_table_counts_every_arm_and_only_gated_arms_pay_for_a_gate(
    tiny: Tiny, tmp_path: Path
) -> None:
    for part in ("corpus", "graphones"):
        shutil.copytree(tiny.root / part, tmp_path / part)
    (tmp_path / "design.json").write_text(json.dumps(sizing_record(tiny.sizing)))
    arms = {name: tiny.raw["arms"][name] for name in ("baseline", "learned-direct")}
    settings = parse({**tiny.raw, "arms": arms})
    inputs = Inputs(settings, tiny.root / "config.json", Layout(tmp_path), "selection")
    tables = read_tables(tmp_path / "graphones")
    batch = counted_batch(inputs, tables)
    found = {arm: cost(inputs, tiny.sizing, tables, batch, arm) for arm in arms}
    assert set(found) == {"baseline", "learned-direct"}
    for item in found.values():
        assert item.train_step > item.forward_word > item.input_word > 0
    assert found["learned-direct"].gate_step > 0 == found["baseline"].gate_step
