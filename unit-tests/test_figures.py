import json
import shutil
from pathlib import Path

import numpy as np
from conftest import Tiny

from data.bilingual import BilingualCorpus
from data.graphones import read_tables
from figures.ambiguity import BINS, ambiguity_bins, binned
from figures.common import Inputs, smoothed
from figures.compute import FLOPS, cost, counted_batch
from figures.losses import BANDS, frequency_bands
from figures.replay import fired
from figures.tables import formatted_cell, macro, numbers, probability, relation
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
    assert relation(0.0001) == "< 0.001"
    assert relation(0.0123) == "= 0.012"
    assert relation(float("nan")) == "--"


def test_an_equivalence_cell_marks_the_equivalence_test_not_the_difference() -> None:
    test = {"mean": 0.004, "low": -0.001, "high": 0.009, "p": 0.03, "n": 5}
    tied = {"test": test, "wins": 3, "equivalence": {"p": 0.2}}
    assert formatted_cell(tied, 0.4) == "$+0.004$ (3)"
    found = {**tied, "equivalence": {"p": 0.01}}
    assert formatted_cell(found, 0.02) == "$\\mathbf{+0.004}^\\dagger$ (3)"
    assert formatted_cell({"test": test, "wins": 3}, None) == "$+0.004^\\dagger$ (3)"


def test_numbers_name_the_pair_and_cover_every_measure_with_wins_and_holm() -> None:
    test = {"mean": 0.025, "low": 0.008, "high": 0.042, "p": 0.009, "n": 10}
    cell = {"test": test, "sign_flip_p": 0.0098, "wins": 8}
    report = {
        "family": {"holm": {"split_vs_bytes/minimal_pairs": 0.035}},
        "contrasts": {"split_vs_bytes": {"minimal_pairs": cell, "bits_per_byte": cell}},
    }
    written = numbers("es-pt", {"graphones-vs-bytes": report})
    name = "EsPtGraphonesVsBytesSplitVsBytes"
    assert f"\\newcommand{{\\{name}MinimalPairsHolm}}{{0.035}}" in written
    assert f"\\newcommand{{\\{name}BitsPerByteWins}}{{8}}" in written
    assert f"{name}BitsPerByteHolm" not in written
    assert (
        f"\\newcommand{{\\{name}MinimalPairsCell}}{{$\\mathbf{{+0.025}}^\\dagger$ (8)}}"
        in written
    )
    assert (
        f"\\newcommand{{\\{name}BitsPerByteCell}}{{$+0.025^\\dagger$ (8)}}" in written
    )


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
    corpus = BilingualCorpus(tmp_path / "corpus")
    found = {
        arm: cost(
            inputs,
            tiny.sizing,
            tables,
            counted_batch(inputs, corpus, tables, arm, FLOPS),
            arm,
        )
        for arm in arms
    }
    assert set(found) == {"baseline", "learned-direct"}
    for item in found.values():
        assert item.train_step > item.forward_word > item.input_word > 0
    assert found["learned-direct"].gate_step > 0 == found["baseline"].gate_step
