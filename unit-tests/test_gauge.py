import pytest
import torch
from hypothesis import given, strategies as st
from strategies import hypothesis_settings
from torch import Tensor

from workers.gauge import variance_share


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
