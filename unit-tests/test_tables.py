import numpy as np
from hypothesis import given, strategies as st
from strategies import hypothesis_settings

from workers.run_tables import fitted_rule, on_sphere


def unit(rows: np.ndarray) -> np.ndarray:
    return rows / np.linalg.norm(rows, axis=-1, keepdims=True)


@given(st.integers(min_value=0, max_value=1000))
@hypothesis_settings
def test_a_fitted_rule_gives_each_row_a_direction(seed: int) -> None:
    rng = np.random.default_rng(seed)
    average = unit(rng.normal(size=(40, 3)))
    rule = fitted_rule(
        average, rng.random((40, 3)), rng.random(40), np.arange(40), 0.01
    )
    assert np.allclose(np.square(rule).sum(1), 1.0)


def test_a_rule_that_is_already_linear_in_the_features_is_recovered_up_to_length() -> (
    None
):
    rng = np.random.default_rng(0)
    features = rng.random((50, 3))
    slope = np.array([[0.3, -0.5, -0.1], [-0.2, 0.4, 0.1], [0.0, 0.3, -0.6]])
    average = np.array([0.5, -0.4, 0.6]) + features @ slope
    rule = fitted_rule(average, features, rng.random(50) + 0.1, np.arange(50), 0.01)
    assert np.allclose(rule, unit(average))


def test_only_the_chosen_rows_shape_the_rule() -> None:
    rng = np.random.default_rng(1)
    features = rng.random((30, 3))
    average = unit(rng.normal(size=(30, 3)))
    rows = np.arange(0, 30, 2)
    changed = average.copy()
    changed[1::2] = unit(rng.normal(size=(15, 3)))
    uses = np.ones(30)
    assert np.allclose(
        fitted_rule(average, features, uses, rows, 0.01),
        fitted_rule(changed, features, uses, rows, 0.01),
    )


def test_heavily_used_rows_pull_the_rule_toward_them() -> None:
    average = np.array([[0.6, -0.2, 0.2]] * 2 + [[0.2, 0.4, -0.4]] * 2)
    uses = np.array([100.0, 100.0, 1.0, 1.0])
    rule = fitted_rule(average, np.zeros((4, 3)), uses, np.arange(4), 0.01)
    assert np.allclose(rule, unit(np.average(average, axis=0, weights=uses)))


def test_a_direction_keeps_its_signs_and_one_too_short_to_trust_is_the_equal_mix() -> (
    None
):
    found = on_sphere(np.array([[0.3, -0.4, 0.0], [0.001, -0.002, 0.0]]), 0.01)
    assert np.allclose(found[0], [0.6, -0.8, 0.0])
    assert np.allclose(found[1], np.full(3, 3**-0.5))
