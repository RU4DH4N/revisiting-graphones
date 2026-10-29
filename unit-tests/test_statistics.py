import math

import numpy as np
import pytest
from hypothesis import given, strategies as st
from scipy import stats
from strategies import hypothesis_settings

from workers.statistics import equivalence, holm, paired, sign_flip_p

finite = st.lists(
    st.floats(min_value=-100, max_value=100, allow_nan=False), min_size=3, max_size=8
)


@given(finite, st.floats(min_value=0.1, max_value=10))
@hypothesis_settings
def test_the_paired_test_matches_scipy(left: list[float], spread: float) -> None:
    right = [v + spread * ((-1) ** k) * (k + 1) for k, v in enumerate(left)]
    cell = paired("minimal_pairs", left, right)
    expected = stats.ttest_rel(left, right)
    interval = expected.confidence_interval(0.95)
    assert cell.test.p == pytest.approx(expected.pvalue, rel=1e-6, abs=1e-9)
    assert cell.test.low == pytest.approx(interval.low, rel=1e-6, abs=1e-9)
    assert cell.test.high == pytest.approx(interval.high, rel=1e-6, abs=1e-9)


def test_lower_bits_per_byte_counts_as_better() -> None:
    cell = paired("bits_per_byte", [1.0, 1.1, 0.9], [1.2, 1.3, 1.25])
    assert cell.test.mean > 0 and cell.wins == 3


def test_a_contrast_uses_only_the_replicates_both_arms_have() -> None:
    cell = paired("minimal_pairs", [1.0] * 10, [0.5, 0.4, 0.6, 0.5, 0.45])
    assert cell.replicates == 5 and len(cell.differences) == 5


def test_a_failed_replicate_is_dropped_and_the_sensitivity_fills_it_against_the_claim() -> (
    None
):
    cell = paired("minimal_pairs", [0.8, 0.7, math.nan], [0.5, 0.6, 0.55])
    assert cell.failed_left == 1 and cell.test.n == 2
    assert cell.sensitivity.n == 3
    assert cell.sensitivity.mean == pytest.approx(((0.3 + 0.1) + (0.7 - 0.55)) / 3)


def test_the_sign_flip_floor_is_reached_only_when_every_seed_agrees() -> None:
    agreeing = np.array([0.1, 0.2, 0.15, 0.3, 0.25])
    assert sign_flip_p(agreeing) == pytest.approx(2 / 32)
    mixed = np.array([0.1, -0.2, 0.15, 0.3, -0.25])
    assert sign_flip_p(mixed) > 2 / 32


def test_equivalence_needs_the_whole_interval_inside_the_seed_spread() -> None:
    right = [0.50, 0.60, 0.40, 0.55, 0.45, 0.52, 0.48, 0.58, 0.42, 0.50]
    close = paired("cka", [v + 0.001 * (k % 2) for k, v in enumerate(right)], right)
    far = paired("cka", [v + 0.3 for v in right], right)
    assert equivalence(close).p < 0.05
    assert equivalence(far).p > 0.5


def test_holm_steps_down_and_never_decreases() -> None:
    found = holm({"a": 0.01, "b": 0.04, "c": 0.03, "d": math.nan})
    assert found["a"] == pytest.approx(0.03)
    assert found["c"] == pytest.approx(0.06)
    assert found["b"] == pytest.approx(0.06)
    assert math.isnan(found["d"])
