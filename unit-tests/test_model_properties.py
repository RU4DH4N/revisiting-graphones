import math
from dataclasses import replace
from functools import partial
from pathlib import Path

import numpy as np
import pytest
import torch
from conftest import Tiny, model_for, model_with, with_arm
from torch import Tensor, nn

from data.splits import SELECTION, TRAIN
from framework.checkpoints import CheckpointError, save_checkpoint
from framework.seeds import seed_for
from models.bilingual import (
    IGNORE,
    RECEIVER_OUTPUTS,
    START,
    GraphoneModel,
    MessageChannel,
)
from models.graphone import GraphoneCode
from models.shared.block import causal_attention
from models.shared.numerics import normalise
from workers import design
from workers.arms import assemble, load_model, mix_spec, stream_tables
from workers.batches import Expander, WindowSampler, WordBatch
from workers.evaluate import Scorer
from workers.train import GATE_PREFIX, train

CPU = torch.device("cpu")


def windows_for(tiny: Tiny, seed: int, windows: int) -> WordBatch:
    rows = WindowSampler(tiny.corpus.parts[TRAIN], [0, 1], 6, windows, seed).draw()
    return Expander(tiny.corpus, tiny.tables.codes, tiny.tables.lengths, CPU)(rows)


def batch_for(tiny: Tiny) -> WordBatch:
    return windows_for(tiny, 0, 4)


def nats_per_symbol(model: GraphoneModel, batch: WordBatch) -> Tensor:
    nll = model.word_nll(batch.inputs, batch.targets, model.messages(batch.context))
    return nll.sum() / batch.targets.ne(IGNORE).sum()


@pytest.mark.parametrize("name", ["baseline", "learned-direct"])
def test_a_message_never_depends_on_later_words_or_other_windows(
    tiny: Tiny, name: str
) -> None:
    model = model_for(tiny, name).eval()
    batch = batch_for(tiny)
    events = batch.context.shape[1]
    t = events // 2
    changed = batch.context.clone()
    changed[0, t] = batch.context[1, 0]
    with torch.no_grad():
        base = model.messages(batch.context)
        base = base.view(-1, events, base.shape[-1])
        after = model.messages(changed).view_as(base)
    torch.testing.assert_close(after[0, :t], base[0, :t])
    torch.testing.assert_close(after[1:], base[1:])
    assert not torch.allclose(after[0, t], base[0, t])


def test_an_untrained_receiver_ignores_the_message_entirely(tiny: Tiny) -> None:
    model = model_for(tiny, "learned-direct").eval()
    batch = batch_for(tiny)
    with torch.no_grad():
        messages = model.messages(batch.context)
        base = model.receiver(batch.inputs, messages)
        torch.testing.assert_close(
            model.receiver(batch.inputs, torch.randn_like(messages)), base
        )


def test_each_spelling_reads_only_its_own_message(tiny: Tiny) -> None:
    model = model_for(tiny, "learned-direct").eval()
    for channel in model.receiver.channels:
        assert isinstance(channel, MessageChannel)
        torch.nn.init.normal_(channel.up.weight)
    batch = batch_for(tiny)
    with torch.no_grad():
        messages = model.messages(batch.context)
        moved = messages.clone()
        moved[0] = torch.randn_like(moved[0])
        base = model.receiver(batch.inputs, messages)
        after = model.receiver(batch.inputs, moved)
    torch.testing.assert_close(after[1:], base[1:])
    assert not torch.allclose(after[0], base[0])


def test_an_untrained_model_spells_at_chance(tiny: Tiny) -> None:
    model = model_for(tiny, "baseline").eval()
    with torch.no_grad():
        start = nats_per_symbol(model, batch_for(tiny)).item()
    assert start == pytest.approx(math.log(RECEIVER_OUTPUTS), abs=0.1)


def test_the_model_can_memorise_one_batch(tiny: Tiny) -> None:
    settings = with_arm(tiny.settings, "learned-direct", hold=False)
    torch.manual_seed(0)
    model = model_with(tiny, settings, "learned-direct").train()
    batch = windows_for(tiny, 0, 2)
    optimiser = torch.optim.Adam(model.parameters(), lr=3e-3)
    loss = nats_per_symbol(model, batch)
    start = loss.item()
    for _ in range(150):
        loss = nats_per_symbol(model, batch)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
    assert loss.item() < 0.2 * start


def test_every_parameter_learns_within_three_steps(tiny: Tiny) -> None:
    settings = with_arm(tiny.settings, "learned-direct", hold=False)
    model = model_with(tiny, settings, "learned-direct").train()
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
    for seed in range(3):
        optimiser.zero_grad()
        nats_per_symbol(model, windows_for(tiny, seed, 4)).backward()
        optimiser.step()
    frozen = [n for n, p in model.named_parameters() if torch.equal(p, before[n])]
    assert frozen == []


def gates(model: GraphoneModel) -> dict[str, Tensor]:
    return {
        name: p.detach().clone()
        for name, p in model.named_parameters()
        if name.startswith(GATE_PREFIX)
    }


@pytest.mark.parametrize("held", [True, False])
def test_a_held_gate_stays_at_its_start_until_release(tiny: Tiny, held: bool) -> None:
    arm = with_arm(tiny.settings, "learned-direct", hold=held)
    settings = replace(arm, gauge=replace(arm.gauge, interval=100))
    trained = train(
        settings,
        replace(tiny.sizing, steps=8),
        tiny.corpus,
        tiny.tables,
        "learned-direct",
        [0],
        [None],
        CPU,
    )[0]
    start, end = (
        gates(model_with(tiny, settings, "learned-direct")),
        gates(trained.model),
    )
    assert trained.metrics.released_at is None
    assert start
    assert all(torch.equal(start[n], end[n]) for n in start) == held


def curve(tiny: Tiny, replicate: int) -> list[float]:
    found = train(
        tiny.settings,
        replace(tiny.sizing, steps=6),
        tiny.corpus,
        tiny.tables,
        "triple-direct",
        [replicate],
        [None],
        CPU,
    )[0]
    return [point.loss for point in found.metrics.curve]


def test_training_is_reproducible_and_seeds_differ(tiny: Tiny) -> None:
    assert curve(tiny, 0) == curve(tiny, 0)
    assert curve(tiny, 0) != curve(tiny, 1)


def gate_start(tiny: Tiny, name: str, replicate: int) -> Tensor:
    model = assemble(tiny.settings, tiny.sizing, tiny.tables, name, replicate)
    return model.trunk.code.gate_hidden.weight.detach()


def test_each_replicate_starts_its_gate_differently_and_every_arm_shares_it(
    tiny: Tiny,
) -> None:
    first = gate_start(tiny, "learned-direct", 0)
    assert torch.equal(first, gate_start(tiny, "learned-direct", 0))
    assert not torch.equal(first, gate_start(tiny, "learned-direct", 1))
    assert torch.equal(first, gate_start(tiny, "learned-direct-matched", 0))


def gate_inputs(tiny: Tiny, replicate: int) -> Tensor:
    settings = with_arm(tiny.settings, "learned-direct", shuffled=True)
    model = assemble(settings, tiny.sizing, tiny.tables, "learned-direct", replicate)
    return model.trunk.code.features


def test_each_replicate_shuffles_the_gate_features_differently(tiny: Tiny) -> None:
    assert torch.equal(gate_inputs(tiny, 0), gate_inputs(tiny, 0))
    assert not torch.equal(gate_inputs(tiny, 0), gate_inputs(tiny, 1))


def test_a_checkpoint_reloads_the_same_model_and_rejects_tampering(
    tiny: Tiny, tmp_path: Path
) -> None:
    model = model_for(tiny, "learned-direct").eval()
    path = tmp_path / "model.pt"
    save_checkpoint(path, model, {"arm": "learned-direct"})
    loaded = load_model(
        path, tiny.settings, tiny.sizing, tiny.tables, "learned-direct", 0
    ).eval()
    batch = batch_for(tiny)
    with torch.no_grad():
        torch.testing.assert_close(
            loaded.messages(batch.context), model.messages(batch.context)
        )
    payload = torch.load(path, weights_only=True)
    first = next(iter(payload["model"]))
    payload["model"][first] = payload["model"][first] + 1.0
    torch.save(payload, path)
    with pytest.raises(CheckpointError, match="does not match the hash"):
        load_model(path, tiny.settings, tiny.sizing, tiny.tables, "learned-direct", 0)


def test_each_word_counts_its_letters_and_stop(tiny: Tiny) -> None:
    batch = windows_for(tiny, 0, 8)
    assert torch.equal(batch.spelled, batch.targets.ne(IGNORE).sum(-1))
    assert batch.inputs.shape[1] == tiny.corpus.max_bytes + 1


def test_explicit_attention_matches_the_library_kernel() -> None:
    torch.manual_seed(0)
    q, k, v = (torch.randn(2, 3, 9, 8) for _ in range(3))
    expected = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
    torch.testing.assert_close(causal_attention(q, k, v), expected)


def test_a_matched_arm_starts_its_words_at_the_projection_arms_scale(
    tiny: Tiny,
) -> None:
    matched = model_for(tiny, "learned-direct-matched")
    projected = model_for(tiny, "baseline")
    context = windows_for(tiny, 0, 16).context
    with torch.no_grad():
        direct = normalise(matched.trunk.code(context), 1e-5) * matched.trunk.scale
        code = normalise(projected.trunk.code(context), 1e-5)
        through = projected.trunk.trunk.in_proj(code)
    assert rms(direct) == pytest.approx(rms(through), rel=0.1)
    assert rms(normalise(matched.trunk.code(context), 1e-5)) > 2 * rms(through)


def rms(values: Tensor) -> float:
    return values.square().mean().sqrt().item()


def test_every_receiver_linear_starts_at_the_initialiser_scale_except_the_zeroed_up(
    tiny: Tiny,
) -> None:
    model = model_for(tiny, "learned-direct")
    std = tiny.sizing.trunk.initialiser_range
    for channel in model.receiver.channels:
        assert isinstance(channel, MessageChannel)
        for layer in (channel.own, channel.sent, channel.down):
            assert abs(layer.weight.std().item() - std) < 0.25 * std
        assert not channel.sent.bias.any() and not channel.up.weight.any()


@pytest.mark.parametrize("name", ["baseline", "learned-direct"])
def test_the_counted_trunk_is_the_real_trunk_without_its_code(
    tiny: Tiny, name: str
) -> None:
    model = model_for(tiny, name)
    shape = tiny.sizing.trunk
    real = sum(
        p.numel()
        for key, p in model.trunk.named_parameters()
        if not key.startswith("code.")
    )
    assert design.shaped_trunk_count(shape, model.trunk.code.dims) == real


def test_the_receiver_predicts_bytes_and_stop_but_never_start(tiny: Tiny) -> None:
    model = model_for(tiny, "baseline")
    batch = batch_for(tiny)
    with torch.no_grad():
        logits = model.receiver(batch.inputs, model.messages(batch.context))
    assert logits.shape[-1] == RECEIVER_OUTPUTS == START


def test_scoring_keeps_every_scored_word_once_and_nothing_else(tiny: Tiny) -> None:
    part = tiny.corpus.parts[SELECTION]
    events = tiny.settings.context_words
    scorer = Scorer(
        model_for(tiny, "baseline"),
        Expander(tiny.corpus, tiny.tables.codes, tiny.tables.lengths, CPU),
        events,
        CPU,
        "fp32",
        0,
    )
    scores = scorer.score(part, [0, 1])
    lengths = np.diff(part.offsets)
    expected = np.concatenate(
        [
            part.ids[part.offsets[document] + 1 : part.offsets[document + 1]]
            for document in range(len(lengths))
        ]
    )
    assert len(scores.types) == lengths.sum() - len(lengths)
    assert sorted(scores.types.tolist()) == sorted(np.asarray(expected).tolist())
    assert (scores.symbols >= 2).all() and (scores.nats > 0).all()


def record_input(seen: list[Tensor], module: nn.Module, args: tuple[Tensor]) -> None:
    seen.append(args[0])


@pytest.mark.parametrize("name", ["baseline", "learned-direct-matched"])
def test_every_word_enters_the_trunk_at_the_same_length(tiny: Tiny, name: str) -> None:
    model = model_for(tiny, name)
    seen: list[Tensor] = []
    model.trunk.trunk.register_forward_pre_hook(partial(record_input, seen))
    with torch.no_grad():
        model.trunk(windows_for(tiny, 0, 16).context)
    lengths = seen[0].norm(dim=-1)
    assert torch.allclose(
        lengths, torch.full_like(lengths, model.trunk.scale), rtol=1e-4
    )


def messages_with_table(tiny: Tiny, table: np.ndarray) -> Tensor:
    model = model_for(tiny, "triple-direct")
    streams = replace(
        stream_tables(tiny.tables, tiny.settings.arms["triple-direct"]), table=table
    )
    spec = mix_spec(tiny.settings, tiny.settings.arms["table-avg"], 0)
    model.trunk.code = GraphoneCode(
        spec, streams, seed_for(tiny.settings.root_seed, 0, "position-codes")
    )
    with torch.no_grad():
        return model.messages(windows_for(tiny, 0, 4).context)


def test_flipping_some_graphones_sound_sign_changes_what_the_trunk_says(
    tiny: Tiny,
) -> None:
    rows = len(tiny.tables.has)
    positive = np.tile(np.sqrt([0.5, 0.3, 0.2]).astype(np.float32), (rows, 1))
    flipped = positive.copy()
    flipped[::2, 1] *= -1
    said = messages_with_table(tiny, positive)
    changed = messages_with_table(tiny, flipped)
    assert not torch.allclose(said, changed, atol=1e-4)
    assert not torch.allclose(said, -changed, atol=1e-4)


def test_an_input_mlp_replaces_only_the_projection(tiny: Tiny) -> None:
    linear = model_for(tiny, "baseline").trunk.trunk
    mlp = model_for(tiny, "baseline-mlp").trunk.trunk
    width = tiny.sizing.trunk.d_model
    assert isinstance(linear.in_proj, nn.Linear)
    assert isinstance(mlp.in_proj, nn.Sequential)
    assert mlp.in_proj[0].out_features == round(tiny.sizing.trunk.mlp_ratio * width)
    assert mlp.in_proj[2].out_features == width
    blocks = [sum(p.numel() for p in t.blocks.parameters()) for t in (linear, mlp)]
    assert blocks[0] == blocks[1]
