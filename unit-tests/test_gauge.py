import math

import pytest
import torch
from hypothesis import given, strategies as st
from strategies import hypothesis_settings
from torch import Tensor

from workers.gauge import direction_share, variance_share


def clustered(
    classes: int, per: int, dims: int, noise: float, seed: int
) -> tuple[Tensor, Tensor]:
    generator = torch.Generator().manual_seed(seed)
    centres = torch.randn(classes, dims, generator=generator) * 5
    labels = torch.arange(classes).repeat_interleave(per)
    values = centres[labels] + noise * torch.randn(
        len(labels), dims, generator=generator
    )
    return values, labels


@given(
    st.integers(min_value=2, max_value=6),
    st.floats(min_value=0.1, max_value=10.0),
    st.floats(min_value=-5.0, max_value=5.0),
    st.integers(min_value=0, max_value=1000),
)
@hypothesis_settings
def test_the_class_share_ignores_scale_offset_and_order(
    classes: int, scale: float, offset: float, seed: int
) -> None:
    values, labels = clustered(classes, 20, 8, 1.0, seed)
    base = variance_share(values, labels, classes)
    order = torch.randperm(len(labels))
    moved = variance_share(values[order] * scale + offset, labels[order], classes)
    assert 0.0 <= base <= 1.0
    assert moved == pytest.approx(base, abs=1e-4)


def test_the_class_share_is_one_when_class_decides_everything_and_near_zero_when_nothing() -> (
    None
):
    values, labels = clustered(4, 50, 8, 0.0, 0)
    assert variance_share(values, labels, 4) == pytest.approx(1.0, abs=1e-6)
    noise = torch.randn(4000, 8, generator=torch.Generator().manual_seed(1))
    random = torch.randint(0, 4, (4000,), generator=torch.Generator().manual_seed(2))
    assert variance_share(noise, random, 4) < 0.01


def test_the_direction_share_is_one_on_a_line_and_small_for_shapeless_values() -> None:
    generator = torch.Generator().manual_seed(3)
    along = torch.randn(200, 1, generator=generator) * torch.randn(
        1, 8, generator=generator
    )
    assert direction_share(along) == pytest.approx(1.0, abs=1e-6)
    shapeless = torch.randn(4000, 8, generator=generator)
    assert direction_share(shapeless) < 0.2


def test_the_direction_share_ignores_scale_and_offset_and_follows_two_groups() -> None:
    values, _ = clustered(2, 100, 8, 1.0, 4)
    base = direction_share(values)
    assert direction_share(values * 3.0 - 2.0) == pytest.approx(base, abs=1e-6)
    assert base > 0.8


@pytest.mark.parametrize("broken", [math.nan, math.inf])
def test_the_direction_share_of_values_that_are_not_finite_is_not_a_number(
    broken: float,
) -> None:
    values = torch.randn(50, 8, generator=torch.Generator().manual_seed(5))
    values[7, 3] = broken
    assert math.isnan(direction_share(values))
