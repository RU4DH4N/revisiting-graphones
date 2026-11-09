import math

import numpy as np
import pytest
import torch

from models.graphone import (
    BYTES,
    EQUAL,
    FEATURES,
    KEPT,
    LEARNED,
    LETTERS,
    MIXED_STREAMS,
    SPLIT,
    TABLE,
    TRIPLE,
    GraphoneCode,
    MixSpec,
    StreamTables,
    gate_features,
)
from models.shared.symbols import BLANK

ROWS = BLANK + 1
DIMS = 8
HAS = np.arange(ROWS) % 4 != 3
HAS[BLANK] = False


def signs(rng: np.random.Generator) -> np.ndarray:
    return rng.choice(np.array([-1, 1], dtype=np.int8), size=(ROWS, DIMS))


def tables(table: np.ndarray | None) -> StreamTables:
    rng = np.random.default_rng(0)
    return StreamTables(
        letters=signs(rng),
        sounds=signs(rng) * HAS[:, None],
        joint=signs(rng),
        has=HAS,
        features=rng.random((ROWS, 3)).astype(np.float32),
        uses=rng.integers(0, 50, ROWS).astype(np.float64),
        table=table,
        slots=5,
    )


def built(
    mix: str, switch: bool, table: np.ndarray | None, streams: str, rotation: bool
) -> GraphoneCode:
    spec = MixSpec(streams, mix, FEATURES, False, switch, 64, 5, 6, rotation, 7)
    return GraphoneCode(spec, tables(table), 11)


def code(mix: str) -> GraphoneCode:
    return built(mix, False, None, TRIPLE, False)


def tabled(table: np.ndarray, switch: bool) -> GraphoneCode:
    return built(TABLE, switch, table, TRIPLE, False)


def fixed(*shares: float) -> np.ndarray:
    return np.tile(np.array([shares], dtype=np.float32), (ROWS, 1))


UNITS = torch.tensor([[0, 1, 3, BLANK, BLANK], [2, 4, 5, 1, BLANK]])


@pytest.mark.parametrize("streams", [LETTERS, BYTES])
def test_a_single_stream_code_reads_no_sounds(streams: str) -> None:
    spec = MixSpec(streams, EQUAL, FEATURES, False, False, 64, 5, 6, False, 7)
    plain = tables(None)
    flipped = StreamTables(
        plain.letters,
        -plain.sounds,
        -plain.joint,
        plain.has,
        plain.features,
        plain.uses,
        None,
        plain.slots,
    )
    found = GraphoneCode(spec, plain, 11)(UNITS)
    assert found.shape[-1] == DIMS
    assert torch.equal(found, GraphoneCode(spec, flipped, 11)(UNITS))


def test_a_kept_mix_starts_equal_and_never_moves_the_letters() -> None:
    found = code(KEPT)
    assert torch.allclose(found(UNITS), code(EQUAL)(UNITS))
    with torch.no_grad():
        found.gate_out.weight.normal_()
    directions = found.directions()
    assert torch.allclose(directions[:, 0], torch.full((ROWS,), MIXED_STREAMS**-0.5))
    assert torch.allclose(directions.norm(dim=-1), torch.ones(ROWS))


def test_a_kept_gate_learns_the_split_of_sound_and_joint() -> None:
    found = code(KEPT)
    optimiser = torch.optim.SGD(found.parameters(), lr=1.0)
    found(UNITS)[..., DIMS:].sum().backward()
    assert found.gate_out.weight.grad is not None
    assert found.gate_out.weight.grad.abs().sum() > 0
    optimiser.step()
    moved = found.directions()
    assert torch.allclose(moved[:, 0], torch.full((ROWS,), MIXED_STREAMS**-0.5))
    assert not torch.allclose(moved[:, 1], moved[:, 2])


def test_a_new_learned_mix_starts_as_the_equal_mix() -> None:
    assert torch.allclose(code(LEARNED)(UNITS), code(EQUAL)(UNITS))


def test_building_a_learned_mix_leaves_the_global_generator_alone() -> None:
    torch.manual_seed(5)
    expected = torch.rand(3)
    torch.manual_seed(5)
    code(LEARNED)
    assert torch.equal(torch.rand(3), expected)


def test_padding_adds_nothing_to_the_word_code() -> None:
    found = code(EQUAL)
    short = torch.tensor([[0, 1, 3]])
    assert torch.allclose(found(UNITS[:1]), found(short))


def test_a_letters_only_share_silences_the_other_streams() -> None:
    found = tabled(fixed(1.0, 0.0, 0.0), False)(torch.tensor([[0, 1]]))
    assert torch.count_nonzero(found[..., DIMS:]) == 0
    assert torch.count_nonzero(found[..., :DIMS]) > 0


def test_a_unit_without_a_sound_is_spelled_by_its_letters_alone() -> None:
    found = tabled(fixed(0.2, 0.4, 0.4), False)(torch.tensor([[3]]))
    letters = tabled(fixed(1.0, 0.0, 0.0), False)(torch.tensor([[3]]))
    assert torch.allclose(found, letters)
    assert torch.count_nonzero(found[..., DIMS:]) == 0


def test_a_switched_table_holds_the_equal_mix_until_release() -> None:
    shares = np.tile(np.array([[0.6, 0.3, 0.1]], dtype=np.float32), (ROWS, 1))
    held = tabled(shares, True)
    assert torch.allclose(held(UNITS), code(EQUAL)(UNITS))
    held.release()
    assert torch.allclose(held(UNITS), tabled(shares, False)(UNITS))
    assert not torch.allclose(held(UNITS), code(EQUAL)(UNITS))


def test_release_survives_a_reload() -> None:
    shares = np.full((ROWS, 3), 1 / 3, dtype=np.float32)
    held = tabled(shares, True)
    held.release()
    fresh = tabled(shares, True)
    fresh.load_state_dict(held.state_dict())
    assert bool(fresh.released)


def test_the_word_code_is_the_slot_sum_of_weighted_streams_times_position_codes() -> (
    None
):
    for streams, mix, table in (
        (SPLIT, EQUAL, None),
        (TRIPLE, EQUAL, None),
        (TRIPLE, TABLE, fixed(0.5, 0.3, 0.2)),
    ):
        found = built(mix, False, table, streams, False)
        parts = [found.letters[UNITS], found.sounds[UNITS]]
        if streams == "triple":
            parts.append(found.joint[UNITS])
        if mix != "equal":
            weights = found.weights()[UNITS]
            parts = [weights[..., k : k + 1] * part for k, part in enumerate(parts)]
        expected = (torch.cat(parts, -1) * found.positions[: UNITS.shape[-1]]).sum(-2)
        torch.testing.assert_close(found(UNITS), expected)


def test_mix_weights_keep_the_total_energy_of_the_streams() -> None:
    weights = tabled(fixed(*np.sqrt([0.6, 0.3, 0.1])), False).weights()
    assert torch.allclose(weights[HAS].square().sum(1), torch.tensor(3.0))
    assert torch.allclose(weights[HAS][:, 0], torch.tensor(math.sqrt(1.8)))


def test_shuffled_features_move_between_sounded_rows_only() -> None:
    rng = np.random.default_rng(3)
    features = rng.random((ROWS, 3)).astype(np.float32)
    found = gate_features(features, HAS, "HGS", True, 9)
    assert np.array_equal(found[~HAS], features[~HAS])
    assert not np.array_equal(found[HAS], features[HAS])
    assert np.array_equal(np.sort(found[HAS], 0), np.sort(features[HAS], 0))


def split_code(mix: str) -> GraphoneCode:
    return built(mix, False, None, SPLIT, False)


def test_a_new_two_stream_learned_mix_starts_as_the_two_stream_equal_mix() -> None:
    assert torch.allclose(split_code(LEARNED)(UNITS), split_code(EQUAL)(UNITS))


def test_two_stream_weights_keep_their_squares_summing_to_two() -> None:
    found = split_code(LEARNED)
    with torch.no_grad():
        found.gate_out.bias.copy_(torch.tensor([1.5, -0.5]))
    weights = found.weights()
    assert weights.shape == (ROWS, 2)
    assert torch.allclose(weights[HAS].square().sum(-1), torch.tensor(2.0))
    assert torch.equal(
        weights[~HAS], torch.tensor([[1.0, 0.0]]).expand(int((~HAS).sum()), 2)
    )


def test_learned_weights_can_be_negative_and_keep_their_squares_summing_to_three() -> (
    None
):
    found = code(LEARNED)
    with torch.no_grad():
        found.gate_out.bias.copy_(torch.tensor([1.0, -2.0, 0.5]))
    weights = found.weights()
    assert bool((weights[HAS][:, 1] < 0).all())
    assert torch.allclose(weights[HAS].square().sum(-1), torch.tensor(3.0))


def test_a_learned_weight_can_cross_zero() -> None:
    found = code(LEARNED)
    with torch.no_grad():
        found.gate_out.bias.copy_(torch.tensor([1.0, 0.0, 1.0]))
    found.weights()[HAS][:, 1].sum().backward()
    gradient = found.gate_out.bias.grad
    assert gradient is not None and float(gradient[1]) > 0


def test_flipping_some_graphones_sound_sign_changes_the_word_code_beyond_a_sign() -> (
    None
):
    positive = fixed(*np.sqrt([0.5, 0.3, 0.2]))
    flipped = positive.copy()
    flipped[::2, 1] *= -1
    words = tabled(positive, False)(UNITS)
    changed = tabled(flipped, False)(UNITS)
    assert not torch.allclose(words, changed)
    assert not torch.allclose(words, -changed)


def rotating() -> GraphoneCode:
    return built(LEARNED, False, None, TRIPLE, True)


def randomised(found: GraphoneCode) -> GraphoneCode:
    generator = torch.Generator().manual_seed(3)
    with torch.no_grad():
        for layer in (found.gate_rotate_angles, found.gate_mirror_out):
            layer.weight.copy_(torch.randn(layer.weight.shape, generator=generator))
    return found


def test_a_new_rotation_turns_nothing_and_mirrors_every_term_alike() -> None:
    words = code(LEARNED)(UNITS)
    mirror = rotating().gate_mirror_out.bias.detach()
    along = (words * mirror).sum(-1, keepdim=True) / (mirror * mirror).sum()
    torch.testing.assert_close(rotating()(UNITS), words - 2 * mirror * along)


def test_rotating_and_mirroring_keep_each_terms_length() -> None:
    found = randomised(rotating())
    terms = torch.randn(
        5, MIXED_STREAMS * DIMS, generator=torch.Generator().manual_seed(4)
    )
    slots = torch.arange(5)
    hidden = found.rotation_hidden(torch.tensor([0, 1, 3, 2, 4]), slots, 4 - slots)
    turned = found.turned(terms, hidden)
    torch.testing.assert_close(turned.norm(dim=-1), terms.norm(dim=-1))
    assert not torch.allclose(turned, terms)


def test_the_rotation_decides_from_the_unit_and_its_slot() -> None:
    found = randomised(rotating())
    slots = torch.arange(3)
    hidden = found.rotation_hidden(torch.tensor([1, 1, 1]), slots, 2 - slots)
    angles = found.gate_rotate_angles(hidden)
    assert not torch.allclose(angles[0], angles[1])
    assert not torch.allclose(angles[1], angles[2])


def test_a_rotated_word_is_the_sum_of_its_turned_slot_terms() -> None:
    found = randomised(rotating())
    expected = []
    for word in UNITS:
        live = word[word != BLANK]
        slots = torch.arange(len(live))
        terms = torch.cat(found.weighted(live), dim=-1)
        hidden = found.rotation_hidden(live, slots, len(live) - 1 - slots)
        turned = found.turned(terms * found.positions[slots], hidden)
        expected.append(turned.sum(0))
    torch.testing.assert_close(found(UNITS), torch.stack(expected))


def test_padding_adds_nothing_to_a_rotated_word_code() -> None:
    found = randomised(rotating())
    short = torch.tensor([[0, 1, 3]])
    torch.testing.assert_close(found(UNITS[:1]), found(short))


def test_building_a_rotation_leaves_the_global_generator_and_the_gate_alone() -> None:
    torch.manual_seed(5)
    expected = torch.rand(3)
    torch.manual_seed(5)
    found = rotating()
    assert torch.equal(torch.rand(3), expected)
    assert torch.equal(found.gate_hidden.weight, code(LEARNED).gate_hidden.weight)
    turned = [
        name
        for name, _ in found.named_parameters()
        if "rotate" in name or "mirror" in name
    ]
    assert turned and all(name.startswith("gate_") for name in turned)
