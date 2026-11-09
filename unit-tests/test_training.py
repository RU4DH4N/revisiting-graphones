import math
import shutil
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import pytest
import torch
from conftest import Tiny, config, model_for, with_arm
from hypothesis import given, strategies as st
from strategies import hypothesis_settings
from torch import nn

from data.graphones import Tables, read_tables, table_file
from framework.runtime import cosine_schedule
from models.graphone import MIXED_STREAMS
from workers import design
from workers.arms import assemble
from workers.records import CurvePoint, GaugePoint
from workers.settings import Config, Release, parse
from workers.stacked import Stack, clip_by_seed
from workers.train import (
    GateAdam,
    Hold,
    Trained,
    optimisers_for,
    prepared,
    run,
    set_rate,
    train,
)

CPU = torch.device("cpu")
OPTIMISER = parse(config()).optimiser


def test_a_held_mix_waits_for_the_language_share_to_rise_and_fall_back() -> None:
    hold = Hold(True, 0.1, True, None)
    assert hold.holding()
    for step, share in enumerate((0.05, 0.04, 0.5, 0.9, 0.85)):
        assert not hold.observe(share, 0.01, step)
    assert hold.holding()
    assert hold.observe(0.8, 0.01, 5)
    assert hold.released_at == 5 and not hold.holding()
    assert not hold.observe(0.1, 0.01, 6) and hold.released_at == 5


def test_a_fall_within_the_gauge_noise_does_not_release() -> None:
    hold = Hold(True, 0.1, True, None)
    for step, share in enumerate((0.1, 0.3, 0.25)):
        assert not hold.observe(share, 0.06, step)
    assert hold.observe(0.2, 0.06, 3)


def test_a_share_that_only_falls_from_its_start_never_releases() -> None:
    hold = Hold(True, 0.1, True, None)
    for step, share in enumerate((0.6, 0.4, 0.2, 0.0)):
        assert not hold.observe(share, 0.0, step)
    assert hold.holding()


def test_the_release_scales_with_the_runs_own_rise() -> None:
    for scale in (0.1, 1.0):
        hold = Hold(True, 0.1, True, None)
        for step, share in enumerate((0.0, 0.9, 0.82)):
            hold.observe(share * scale, 0.0, step)
        assert hold.released_at is None
        assert hold.observe(0.8 * scale, 0.0, 3)


def test_an_unheld_mix_learns_from_the_start() -> None:
    hold = Hold(False, 0.1, False, None)
    assert not hold.holding()
    assert not hold.observe(0.9, 0.0, 1) and not hold.observe(0.1, 0.0, 2)


def stack_for(tiny: Tiny, name: str, seeds: int) -> Stack:
    return Stack(
        [
            assemble(tiny.settings, tiny.sizing, tiny.tables, name, r)
            for r in range(seeds)
        ]
    )


def test_the_gate_has_its_own_optimiser_and_rate(tiny: Tiny) -> None:
    stack = stack_for(tiny, "learned-direct", 2)
    optimiser, gates = optimisers_for(stack, 0.01, 20.0, OPTIMISER, 2, CPU)
    code = stack.template.trunk.code
    learned = {id(p) for p in code.parameters()}
    names = [n for n, p in stack.template.trunk.named_parameters() if id(p) in learned]
    expected = {id(stack.trunk[name]) for name in names}
    assert expected and {id(p) for p in gates.gates} == expected
    trained = {id(p) for group in optimiser.param_groups for p in group["params"]}
    assert not trained & expected
    set_rate(optimiser, gates, 0.01, 50, 100, 0)
    assert gates.lr == pytest.approx(20.0 * optimiser.param_groups[0]["lr"])


def test_the_gate_optimiser_is_adamw_started_when_each_seed_starts_learning() -> None:
    torch.manual_seed(0)
    starts = [0, 3, 6]
    shape = (4, 5)
    initial = torch.randn(len(starts), *shape)
    grads = torch.randn(9, len(starts), *shape)
    stacked = initial.clone().requires_grad_()
    gates = GateAdam([stacked], len(starts), 20.0, (0.9, 0.999), 1e-8, CPU)
    alone = [initial[k].clone().requires_grad_() for k in range(len(starts))]
    references = [torch.optim.AdamW([p], lr=0.0, weight_decay=0.0) for p in alone]
    for index, grad in enumerate(grads):
        rate = 0.01 * (1.0 - index / len(grads))
        gates.lr = 20.0 * rate
        learning = torch.tensor([index >= start for start in starts])
        stacked.grad = grad.clone()
        gates.step(learning)
        for k, (p, reference) in enumerate(zip(alone, references, strict=True)):
            if learning[k]:
                p.grad = grad[k].clone()
                reference.param_groups[0]["lr"] = 20.0 * rate
                reference.step()
        for k, p in enumerate(alone):
            torch.testing.assert_close(stacked.detach()[k], p.detach())
    assert all(not torch.equal(stacked[k], initial[k]) for k in range(len(starts)))


def test_only_per_seed_matrices_decay(tiny: Tiny) -> None:
    stack = stack_for(tiny, "learned-direct", 2)
    found = optimisers_for(stack, 0.01, 20.0, OPTIMISER, 2, CPU)
    decayed, plain = found[0].param_groups[:2]
    names = {id(p): n for n, p in stack.trunk.items()}
    names.update({id(p): n for n, p in stack.receiver.items()})
    template = {
        **dict(stack.template.trunk.named_parameters()),
        **dict(stack.template.receiver.named_parameters()),
    }
    assert decayed["weight_decay"] > 0.0 and plain["weight_decay"] == 0.0
    assert all(template[names[id(p)]].ndim >= 2 for p in decayed["params"])
    assert all(template[names[id(p)]].ndim < 2 for p in plain["params"])


def test_an_equal_mix_has_no_gates(tiny: Tiny) -> None:
    stack = stack_for(tiny, "triple-direct", 1)
    assert optimisers_for(stack, 0.01, 20.0, OPTIMISER, 1, CPU)[1].gates == []


def test_each_seed_is_clipped_by_its_own_norm() -> None:
    torch.manual_seed(0)
    grads = [torch.randn(3, 5, 4) * torch.tensor([0.01, 1.0, 50.0])[:, None, None]]
    grads.append(torch.randn(3, 7) * torch.tensor([0.01, 1.0, 50.0])[:, None])
    expected = []
    for seed in range(3):
        own = [nn.Parameter(torch.zeros_like(g[seed])) for g in grads]
        for p, g in zip(own, grads, strict=True):
            p.grad = g[seed].clone()
        nn.utils.clip_grad_norm_(own, 1.0)
        expected.append([p.grad for p in own])
    clip_by_seed(grads, 1.0)
    for seed in range(3):
        for g, e in zip(grads, expected[seed], strict=True):
            torch.testing.assert_close(g[seed], e)


def tables_with_avg(tiny: Tiny, root: Path) -> Tables:
    shutil.copytree(tiny.root / "graphones", root / "graphones")
    rng = np.random.default_rng(0)
    shares = rng.dirichlet(np.ones(3), len(tiny.tables.has)).astype(np.float32)
    np.save(root / "graphones" / table_file("avg"), shares)
    return read_tables(root / "graphones")


def test_releasing_one_seed_leaves_the_others_held(tiny: Tiny, tmp_path: Path) -> None:
    settings = with_arm(tiny.settings, "table-avg", hold=True)
    tables = tables_with_avg(tiny, tmp_path)
    stack = Stack(
        [assemble(settings, tiny.sizing, tables, "table-avg", r) for r in range(3)]
    )
    assert stack.released.tolist() == [False, False, False]
    stack.release(1)
    assert stack.released.tolist() == [False, True, False]
    assert stack.model(1).trunk.code.released.item()
    assert not stack.model(2).trunk.code.released.item()


@pytest.mark.parametrize(
    ("name", "fall"),
    [
        ("learned-direct", 0.01),
        ("learned-direct", None),
        ("baseline", None),
        ("table-avg", None),
    ],
)
def test_seeds_trained_together_match_each_seed_trained_alone(
    tiny: Tiny, tmp_path: Path, name: str, fall: float | None
) -> None:
    seeded = with_arm(tiny.settings, name, seeds=3)
    held = with_arm(seeded, name, hold=True)
    settings = seeded if fall is None else replace(held, hold=Release(fall))
    sizing = replace(tiny.sizing, steps=7)
    tables = tables_with_avg(tiny, tmp_path)
    together = train(
        settings, sizing, tiny.corpus, tables, name, [0, 1, 2], [None] * 3, CPU
    )
    for seed, joint in enumerate(together):
        (alone,) = train(
            settings, sizing, tiny.corpus, tables, name, [seed], [None], CPU
        )
        assert joint.metrics.released_at == alone.metrics.released_at
        assert_same_points(joint.metrics.curve, alone.metrics.curve)
        assert_same_points(joint.metrics.gauge, alone.metrics.gauge)
        assert_same_weights(joint.model, alone.model)


def assert_same_points(
    left: list[CurvePoint] | list[GaugePoint],
    right: list[CurvePoint] | list[GaugePoint],
) -> None:
    for a, b in zip(left, right, strict=True):
        found, other = asdict(a), asdict(b)
        assert found.pop("step") == other.pop("step")
        found.pop("seconds", None)
        other.pop("seconds", None)
        np.testing.assert_allclose(
            np.hstack(list(found.values())),
            np.hstack(list(other.values())),
            rtol=1e-4,
            atol=1e-6,
        )


def assert_same_weights(left: nn.Module, right: nn.Module) -> None:
    for (n, p), q in zip(
        left.state_dict().items(), right.state_dict().values(), strict=True
    ):
        torch.testing.assert_close(p, q, rtol=1e-4, atol=1e-5, msg=n)


def test_without_the_projection_the_code_feeds_the_trunk_directly(tiny: Tiny) -> None:
    direct = model_for(tiny, "learned-direct")
    projected = model_for(tiny, "baseline")
    assert isinstance(direct.trunk.trunk.in_proj, nn.Identity)
    assert direct.trunk.trunk.ln_f.weight.shape == (3 * 16,)
    assert projected.trunk.trunk.in_proj.weight.shape == (3 * 16, 2 * 32)


def test_a_larger_receiver_keeps_the_message_and_grows(tiny: Tiny) -> None:
    settings = tiny.settings
    sizes = [
        design.receiver_count(settings, tiny.corpus.max_bytes, scale)
        for scale in (1, 2, 4)
    ]
    shapes = [
        design.receiver_for(settings, tiny.corpus.max_bytes, scale)
        for scale in (1, 2, 4)
    ]
    assert sizes[0] < sizes[1] < sizes[2]
    assert len({shape.message_dims for shape in shapes}) == 1


def test_a_short_run_learns_and_records_its_gauge(tiny: Tiny) -> None:
    trained = train(
        tiny.settings,
        replace(tiny.sizing, steps=12),
        tiny.corpus,
        tiny.tables,
        "learned-direct",
        [0],
        [None],
        CPU,
    )[0]
    metrics = trained.metrics
    assert metrics.steps == 12
    assert all(math.isfinite(point.loss) for point in metrics.curve)
    assert [point.step for point in metrics.gauge] == [1, 5, 10]
    steps, table = trained.gates.snapshot_steps[0], trained.gates.snapshots[0]
    assert steps == 0 and table.shape == (len(tiny.tables.has), MIXED_STREAMS)
    assert np.allclose(np.square(table).sum(1), 1.0)


def test_every_step_is_recorded_and_the_curve_is_its_block_average(
    tiny: Tiny,
) -> None:
    trained = train(
        tiny.settings,
        replace(tiny.sizing, steps=12),
        tiny.corpus,
        tiny.tables,
        "learned-direct",
        [0, 1],
        [None, None],
        CPU,
    )
    for found in trained:
        steps = found.steps
        assert steps.loss.shape == (12,)
        assert steps.nats.shape == steps.symbols.shape == (12, 2)
        ends = [point.step for point in found.metrics.curve]
        assert ends == [5, 10, 12]
        for start, point in zip([0, *ends], found.metrics.curve, strict=False):
            block = slice(start, point.step)
            nats = steps.nats[block].sum(0) / steps.symbols[block].sum(0)
            assert point.loss == pytest.approx(steps.loss[block].mean(), rel=1e-5)
            assert point.nats_per_symbol == pytest.approx(nats.tolist(), rel=1e-5)


def releases(readings: list[tuple[float, float]], fall: float, step: int) -> bool:
    shares = [share for share, _ in readings]
    share, noise = readings[step]
    peak = max(shares[: step + 1])
    rise, fallen = peak - shares[0], peak - share
    return rise > noise and fallen > noise and fallen >= fall * rise


def first_release(readings: list[tuple[float, float]], fall: float) -> int | None:
    found = (s for s in range(len(readings)) if releases(readings, fall, s))
    return next(found, None)


@given(
    st.lists(
        st.tuples(
            st.floats(min_value=0.0, max_value=1.0),
            st.floats(min_value=0.0, max_value=0.2),
        ),
        max_size=30,
    ),
    st.floats(min_value=0.01, max_value=0.99),
)
@hypothesis_settings
def test_the_hold_releases_at_the_first_step_a_brute_force_check_would(
    readings: list[tuple[float, float]], fall: float
) -> None:
    hold = Hold(True, fall, True, None)
    for step, (share, noise) in enumerate(readings):
        hold.observe(share, noise, step)
    assert hold.released_at == first_release(readings, fall)


def test_each_gate_snapshot_step_is_recorded_once(tiny: Tiny) -> None:
    trained = train(
        replace(tiny.settings, snapshot_interval=5),
        replace(tiny.sizing, steps=11),
        tiny.corpus,
        tiny.tables,
        "learned-direct",
        [0],
        [None],
        CPU,
    )[0]
    assert trained.gates.snapshot_steps.tolist() == [0, 5, 10]


@dataclass
class ScriptedHold(Hold):
    at: int | None

    def observe(self, share: float, noise: float, step: int) -> bool:
        released = self.holding() and step == self.at
        if released:
            self.released_at = step
        return released


def scripted(
    tiny: Tiny, settings: Config, replicates: list[int], at: list[int | None]
) -> list[Trained]:
    sizing = replace(tiny.sizing, steps=7)
    ready = prepared(
        settings,
        sizing,
        tiny.corpus,
        tiny.tables,
        "learned-direct",
        replicates,
        [None for _ in replicates],
        CPU,
    )
    holds: list[Hold] = [ScriptedHold(True, 0.01, True, None, step) for step in at]
    ready.session.holds = holds
    return run(ready)


def assert_frozen_until_release(found: Trained, steps: int) -> None:
    release = found.metrics.released_at or steps
    release = steps if release == steps - 1 else release
    gates = found.gates
    start = gates.snapshots[0]
    for step, table in zip(gates.snapshot_steps, gates.snapshots, strict=True):
        assert np.array_equal(table, start) == (step <= release), (release, step)


def test_seeds_that_release_at_different_steps_still_match_training_alone(
    tiny: Tiny,
) -> None:
    held = with_arm(tiny.settings, "learned-direct", seeds=3, hold=True)
    settings = replace(held, snapshot_interval=1, gauge=replace(held.gauge, interval=1))
    at = [2, 4, None]
    together = scripted(tiny, settings, [0, 1, 2], at)
    assert [found.metrics.released_at for found in together] == at
    for found in together:
        assert_frozen_until_release(found, 7)
    for seed, joint in enumerate(together):
        (alone,) = scripted(tiny, settings, [seed], [at[seed]])
        assert joint.metrics.released_at == alone.metrics.released_at
        assert_same_weights(joint.model, alone.model)


def test_the_schedule_warms_up_from_the_first_step_and_ends_at_zero() -> None:
    steps, warmup = 100, 10
    rates = [cosine_schedule(step, steps, warmup) for step in range(steps)]
    assert rates[0] == pytest.approx(0.1)
    assert rates[warmup - 1] == pytest.approx(1.0)
    assert rates[warmup] == pytest.approx(1.0)
    assert rates[-1] == pytest.approx(0.0)
    assert all(a >= b for a, b in zip(rates[warmup:], rates[warmup + 1 :]))


def test_a_fixed_release_fires_at_its_step_and_ignores_the_gauge() -> None:
    hold = Hold(True, 0.1, False, 4)
    for step, share in enumerate((0.05, 0.9, 0.1)):
        assert not hold.observe(share, 0.0, step)
    assert not hold.reach(3) and hold.holding()
    assert hold.reach(4) and hold.released_at == 4
    assert not hold.reach(5) and hold.released_at == 4


def test_a_hold_with_no_rule_and_no_step_never_releases() -> None:
    hold = Hold(True, 0.1, False, None)
    for step, share in enumerate((0.0, 0.9, 0.1)):
        assert not hold.observe(share, 0.0, step) and not hold.reach(step)
    assert hold.holding()


def test_a_fixed_arm_releases_exactly_at_its_step_and_holds_until_then(
    tiny: Tiny,
) -> None:
    fixed = with_arm(tiny.settings, "learned-direct", release_fraction=0.25)
    settings = replace(fixed, snapshot_interval=1)
    found = train(
        settings,
        replace(tiny.sizing, steps=12),
        tiny.corpus,
        tiny.tables,
        "learned-direct",
        [0],
        [3],
        CPU,
    )[0]
    assert found.metrics.released_at == 3
    assert_frozen_until_release(found, 12)


def test_a_checkpoint_holds_the_model_of_its_step(tiny: Tiny) -> None:
    unheld = with_arm(tiny.settings, "learned-direct", hold=False)
    settings = replace(unheld, snapshot_interval=6)
    found = train(
        settings,
        replace(tiny.sizing, steps=12),
        tiny.corpus,
        tiny.tables,
        "learned-direct",
        [0],
        [None],
        CPU,
    )[0]
    assert list(found.checkpoints) == [0.5]
    halfway = found.checkpoints[0.5].trunk.code
    with torch.no_grad():
        directions = halfway.directions().numpy()
    snapshot = found.gates.snapshots[list(found.gates.snapshot_steps).index(6)]
    assert np.array_equal(directions, snapshot)
    assert not np.array_equal(directions, found.gates.final)


def test_a_stack_gives_each_seed_its_own_shuffled_features(tiny: Tiny) -> None:
    settings = with_arm(tiny.settings, "learned-direct", shuffled=True)
    models = [
        assemble(settings, tiny.sizing, tiny.tables, "learned-direct", r)
        for r in range(2)
    ]
    weight = torch.randn(models[0].trunk.code.gate_out.weight.shape)
    with torch.no_grad():
        for model in models:
            model.trunk.code.gate_out.weight.copy_(weight)
    stack = Stack(models)
    features = models[0].trunk.code.features
    assert [n for n, b in models[0].trunk.named_buffers() if b is features][0] in (
        stack.buffers
    )
    context = torch.from_numpy(tiny.tables.codes[:, :6].astype(np.int64))[:, None]
    with torch.no_grad():
        stacked = stack.messages(stack.trunk_state(), context)
        for k, model in enumerate(models):
            torch.testing.assert_close(stacked[k], model.messages(context[k]))
