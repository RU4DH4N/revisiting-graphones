import pytest
import torch
from hypothesis import given, strategies as st
from strategies import hypothesis_settings

from models.shared.block import Block
from models.shared.rope import RotaryEmbedding
from models.shared.swiglu import swiglu_hidden_dim
from models.shared.trunk import Shape, Trunk


@st.composite
def trunk_shapes(draw: st.DrawFn) -> tuple[int, int, int]:
    head_dim = draw(st.sampled_from([8, 16, 32]))
    n_heads = draw(st.integers(min_value=1, max_value=4))
    n_layers = draw(st.integers(min_value=1, max_value=3))
    return head_dim * n_heads, n_heads, n_layers


def build(d_model: int, n_heads: int, n_layers: int) -> Trunk:
    return shaped(d_model, n_heads, n_layers, 64, 32, 64)


def shaped(
    d_model: int,
    n_heads: int,
    n_layers: int,
    in_dims: int,
    out_dims: int,
    max_seq_len: int,
) -> Trunk:
    shape = Shape(
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        mlp_ratio=4.0,
        multiple_of=32,
        max_seq_len=max_seq_len,
        rope_theta=10_000.0,
        norm_eps=1e-6,
        initialiser_range=0.02,
        message_dims=out_dims,
    )
    return Trunk(shape, in_dims, out_dims).eval()


@given(trunk_shapes(), st.integers(min_value=2, max_value=16), st.data())
@hypothesis_settings
def test_a_position_sees_itself_and_its_past_but_never_its_future(
    shape: tuple[int, int, int], T: int, data: st.DataObject
) -> None:
    trunk = build(*shape)
    t = data.draw(st.integers(min_value=0, max_value=T - 1))
    x = torch.randn(1, T, 64)
    bumped = x.clone()
    bumped[:, t] += 50.0
    with torch.no_grad():
        base, out = trunk(x), trunk(bumped)
    torch.testing.assert_close(out[:, :t], base[:, :t], rtol=0.0, atol=1e-4)
    assert not torch.allclose(out[:, t], base[:, t], atol=1e-5)


@given(trunk_shapes(), st.integers(min_value=2, max_value=6))
@hypothesis_settings
def test_windows_in_a_batch_never_see_each_other(
    shape: tuple[int, int, int], B: int
) -> None:
    trunk = build(*shape)
    x = torch.randn(B, 10, 64)
    changed = x.clone()
    changed[0] = torch.randn(10, 64)
    perm = torch.randperm(B)
    with torch.no_grad():
        base = trunk(x)
        torch.testing.assert_close(trunk(changed)[1:], base[1:])
        torch.testing.assert_close(trunk(x[perm]), base[perm])


def test_residual_projections_start_smaller_with_depth() -> None:
    torch.manual_seed(7)
    trunk = build(16, 2, 2)
    expected = 0.02 / (2 * 2) ** 0.5
    for block in trunk.blocks:
        assert isinstance(block, Block)
        assert abs(block.out_proj.weight.std().item() - expected) < 0.003
        assert abs(block.mlp.down_proj.weight.std().item() - expected) < 0.003


def test_matching_widths_skip_the_seams_and_others_project_without_bias() -> None:
    same = shaped(16, 2, 1, 16, 16, 64)
    assert isinstance(same.in_proj, torch.nn.Identity)
    assert isinstance(same.out, torch.nn.Identity)
    other = shaped(16, 2, 1, 8, 4, 64)
    assert isinstance(other.in_proj, torch.nn.Linear) and other.in_proj.bias is None
    assert isinstance(other.out, torch.nn.Linear) and other.out.bias is None


def test_the_positional_limit_is_inclusive_and_enforced() -> None:
    trunk = shaped(64, 4, 1, 64, 32, 16)
    assert trunk(torch.randn(1, 16, 64)).shape == (1, 16, 32)
    with pytest.raises(
        ValueError, match="sequence length 17 exceeds the configured 16"
    ):
        trunk(torch.randn(1, 17, 64))


@pytest.mark.parametrize(
    ("d_model", "n_heads", "message"),
    [(100, 3, "not divisible"), (64, 0, "n_heads must be positive"), (30, 2, "even")],
)
def test_impossible_head_layouts_are_rejected(
    d_model: int, n_heads: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        build(d_model, n_heads, 1)


@given(st.sampled_from([8, 16, 32]), st.integers(min_value=4, max_value=24), st.data())
@hypothesis_settings
def test_rotary_scores_depend_only_on_relative_offset(
    head_dim: int, T: int, data: st.DataObject
) -> None:
    rope = RotaryEmbedding(head_dim, T, 10_000.0)
    q = torch.randn(head_dim).expand(1, 1, T, head_dim).contiguous()
    k = torch.randn(head_dim).expand(1, 1, T, head_dim).contiguous()
    qr, kr = rope(torch.stack((q, k)))
    m = data.draw(st.integers(min_value=0, max_value=T - 2))
    n = data.draw(st.integers(min_value=0, max_value=T - 2))
    shift = data.draw(st.integers(min_value=1, max_value=T - 1 - max(m, n)))
    torch.testing.assert_close(
        (qr[0, 0, m] * kr[0, 0, n]).sum(),
        (qr[0, 0, m + shift] * kr[0, 0, n + shift]).sum(),
        rtol=0.0,
        atol=1e-4,
    )
    torch.testing.assert_close(qr.norm(dim=-1), q.norm(dim=-1), rtol=1e-4, atol=1e-4)


@given(
    st.integers(min_value=16, max_value=1024),
    st.floats(min_value=0.5, max_value=8.0),
    st.sampled_from([8, 32, 64]),
)
@hypothesis_settings
def test_the_mlp_width_is_the_smallest_aligned_width_at_two_thirds_the_ratio(
    d_model: int, ratio: float, multiple_of: int
) -> None:
    width = swiglu_hidden_dim(d_model, ratio, multiple_of)
    target = d_model * ratio * 2 / 3
    assert width % multiple_of == 0
    assert target - 1e-6 <= width < target + multiple_of
