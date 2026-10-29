import contextlib
import copy
import math
import time
from collections.abc import Iterator
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import Tensor

from data.bilingual import BilingualCorpus
from data.graphones import Tables
from data.splits import TRAIN
from framework.runtime import (
    adamw,
    autocast_context,
    cosine_schedule,
    device_name,
    on_cuda,
    set_seed,
)
from framework.seeds import seed_for
from models.bilingual import GraphoneModel
from models.graphone import EQUAL
from workers.arms import assemble
from workers.batches import (
    Expander,
    Rows,
    SeedSamplers,
    WindowSampler,
    WordBatch,
    prefetched,
)
from workers.design import Sizing
from workers.gauge import variance_share
from workers.records import (
    CurvePoint,
    GateRecord,
    GaugePoint,
    Metrics,
    StepRecord,
    no_gates,
)
from workers.settings import Config, Optimiser
from workers.stacked import CODE, Stack, clip_by_seed

GATE_PREFIX = CODE + "gate_"


@dataclass
class Hold:
    held: bool
    fall: float
    ruled: bool
    fixed: int | None
    start: float | None = field(default=None, init=False)
    peak: float = field(default=0.0, init=False)
    released_at: int | None = field(default=None, init=False)

    def holding(self) -> bool:
        return self.held and self.released_at is None

    def reach(self, step: int) -> bool:
        if self.holding() and self.fixed is not None and step >= self.fixed:
            self.released_at = step
            return True
        return False

    def observe(self, share: float, noise: float, step: int) -> bool:
        if not self.holding() or not self.ruled:
            return False
        if self.start is None:
            self.start = share
        self.peak = max(self.peak, share)
        rise = self.peak - self.start
        fallen = self.peak - share
        if rise > noise and fallen > noise and fallen >= self.fall * rise:
            self.released_at = step
        return self.released_at is not None


# https://arxiv.org/abs/1412.6980
class GateAdam:
    def __init__(
        self,
        gates: list[Tensor],
        seeds: int,
        scale: float,
        betas: tuple[float, float],
        eps: float,
        device: torch.device,
    ) -> None:
        self.gates = gates
        self.scale = scale
        self.betas = betas
        self.eps = eps
        self.lr = 0.0
        self.first = [torch.zeros_like(p) for p in gates]
        self.second = [torch.zeros_like(p) for p in gates]
        self.steps = torch.zeros(seeds, device=device)

    @torch.no_grad()
    def step(self, learning: Tensor) -> None:
        if not self.gates:
            return
        beta1, beta2 = self.betas
        self.steps += learning.float()
        counted = self.steps.clamp_min(1.0)
        first_fix = 1.0 - beta1**counted
        second_fix = (1.0 - beta2**counted).sqrt()
        for p, first, second in zip(self.gates, self.first, self.second, strict=True):
            self.update(p, first, second, learning, first_fix, second_fix)

    def update(
        self,
        p: Tensor,
        first: Tensor,
        second: Tensor,
        learning: Tensor,
        first_fix: Tensor,
        second_fix: Tensor,
    ) -> None:
        beta1, beta2 = self.betas
        shape = (-1, *(1,) * (p.ndim - 1))
        live = learning.view(shape)
        grad = torch.zeros_like(p) if p.grad is None else p.grad
        first.copy_(torch.where(live, first.lerp(grad, 1.0 - beta1), first))
        moved = second * beta2 + (1.0 - beta2) * grad * grad
        second.copy_(torch.where(live, moved, second))
        denom = second.sqrt() / second_fix.view(shape) + self.eps
        update = self.lr * (first / first_fix.view(shape)) / denom
        p.sub_(torch.where(live, update, 0.0))
        p.grad = None


def set_rate(
    optimiser: torch.optim.Optimizer,
    gates: GateAdam,
    rate: float,
    step: int,
    steps: int,
    warmup: int,
) -> None:
    lr = rate * cosine_schedule(step, steps, warmup)
    for group in optimiser.param_groups:
        group["lr"] = lr
    gates.lr = gates.scale * lr


def optimisers_for(
    stack: Stack,
    rate: float,
    gate_rate: float,
    settings: Optimiser,
    seeds: int,
    device: torch.device,
) -> tuple[torch.optim.AdamW, GateAdam]:
    named = stack.params.items()
    rest = [p for name, p in named if not name.startswith(GATE_PREFIX)]
    optimiser = adamw(
        [p for p in rest if p.ndim > 2],
        [p for p in rest if p.ndim <= 2],
        rate,
        settings.weight_decay,
        settings.betas,
        settings.eps,
        device,
    )
    gates = [p for name, p in named if name.startswith(GATE_PREFIX)]
    return optimiser, GateAdam(
        gates, seeds, gate_rate, settings.betas, settings.eps, device
    )


class Accumulator:
    def __init__(self, seeds: int, languages: int, device: torch.device) -> None:
        self.seeds = seeds
        self.languages = languages
        self.device = device
        self.offsets = torch.arange(seeds, device=device)[:, None] * languages
        self.reset()

    def reset(self) -> None:
        self.loss = torch.zeros(self.seeds, device=self.device)
        self.steps = 0
        self.nats = torch.zeros(self.seeds * self.languages, device=self.device)
        self.symbols = torch.zeros(self.seeds * self.languages, device=self.device)

    def add(self, loss: Tensor, words: Tensor, batch: WordBatch) -> None:
        self.loss += loss.detach()
        self.steps += 1
        slots = (self.offsets + batch.languages).reshape(-1)
        self.nats.index_add_(0, slots, words.detach().reshape(-1))
        self.symbols.index_add_(0, slots, batch.spelled.float().reshape(-1))

    def points(self, step: int, seconds: float) -> list["CurvePoint"]:
        loss = (self.loss / max(self.steps, 1)).tolist()
        rates = (self.nats / self.symbols.clamp_min(1)).view(self.seeds, -1).tolist()
        return [
            CurvePoint(step, seconds, value, rate)
            for value, rate in zip(loss, rates, strict=True)
        ]


class StepLog:
    def __init__(
        self, steps: int, seeds: int, languages: int, device: torch.device
    ) -> None:
        self.languages = languages
        self.offsets = torch.arange(seeds, device=device)[:, None] * languages
        self.loss = torch.zeros(steps, seeds, device=device)
        self.nats = torch.zeros(steps, seeds * languages, device=device)
        self.symbols = torch.zeros(steps, seeds * languages, device=device)

    def add(self, index: int, loss: Tensor, words: Tensor, batch: WordBatch) -> None:
        slots = (self.offsets + batch.languages).reshape(-1)
        self.loss[index] = loss.detach()
        self.nats[index].index_add_(0, slots, words.detach().reshape(-1))
        self.symbols[index].index_add_(0, slots, batch.spelled.float().reshape(-1))

    def record(self, seed: int) -> StepRecord:
        columns = slice(seed * self.languages, (seed + 1) * self.languages)
        return StepRecord(
            self.loss[:, seed].cpu().numpy(),
            self.nats[:, columns].cpu().numpy(),
            self.symbols[:, columns].cpu().numpy(),
        )


@dataclass(frozen=True)
class Trained:
    model: GraphoneModel
    metrics: Metrics
    gates: GateRecord
    steps: StepRecord
    checkpoints: dict[float, GraphoneModel]


def gate_table(model: GraphoneModel) -> np.ndarray:
    code = model.trunk.code
    with torch.no_grad():
        return code.directions().float().cpu().numpy()


class GateHistory:
    def __init__(self, seeds: int, mixed: bool) -> None:
        self.mixed = mixed
        self.steps: list[int] = []
        self.tables: list[list[np.ndarray]] = [[] for _ in range(seeds)]

    def take(self, stack: Stack, step: int) -> None:
        self.steps.append(step)
        for k, tables in enumerate(self.tables):
            tables.append(gate_table(stack.model(k)))

    def record(self, seed: int, model: GraphoneModel) -> GateRecord:
        return (
            GateRecord(
                gate_table(model),
                np.array(self.steps, dtype=np.int64),
                np.stack(self.tables[seed]),
            )
            if self.mixed
            else no_gates()
        )


def reading(messages: Tensor, groups: Tensor, classes: int, step: int) -> GaugePoint:
    half = messages.shape[0] // 2
    halves = [
        variance_share(messages[part], groups[part], classes)
        for part in (slice(None, half), slice(half, None))
    ]
    share = variance_share(messages, groups, classes)
    return GaugePoint(step, share, abs(halves[0] - halves[1]))


class LanguageGauge:
    def __init__(
        self,
        batches: list[WordBatch],
        classes: int,
        precision: str,
        device: torch.device,
    ) -> None:
        self.batches = batches
        self.classes = classes
        self.precision = precision
        self.device = device
        self.groups = torch.cat([b.languages for b in batches], dim=1)

    def messages(self, stack: Stack) -> Tensor:
        with torch.no_grad(), autocast_context(self.device, self.precision):
            found = [
                stack.messages(stack.state(), b.context).to(torch.float32, copy=True)
                for b in self.batches
            ]
        return torch.cat(found, dim=1)

    def read(self, stack: Stack, step: int) -> list[GaugePoint]:
        messages = self.messages(stack)
        return [
            reading(messages[k], self.groups[k], self.classes, step)
            for k in range(len(messages))
        ]


def losses(
    stack: Stack, state: dict[str, Tensor], batch: WordBatch
) -> tuple[Tensor, Tensor]:
    messages = stack.messages(state, batch.context)
    words = stack.word_nll(state, batch.inputs, batch.targets, messages)
    words = words * batch.scored.float()
    return words.sum(1) / batch.spelled.sum(1).clamp_min(1), words


def step(
    stack: Stack,
    optimiser: torch.optim.Optimizer,
    gates: GateAdam,
    learning: Tensor,
    batch: WordBatch,
    precision: str,
    clip: float,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    with autocast_context(device, precision):
        loss, words = losses(stack, stack.state(), batch)
    loss.sum().backward()
    clip_by_seed([p.grad for p in stack.params.values() if p.grad is not None], clip)
    optimiser.step()
    optimiser.zero_grad(set_to_none=True)
    gates.step(learning)
    return loss, words


def learning_mask(holds: list[Hold], device: torch.device) -> Tensor:
    return torch.tensor([not hold.holding() for hold in holds], device=device)


def compile_for(stack: Stack, device: torch.device) -> None:
    if on_cuda(device):
        for value in stack.state().values():
            torch._dynamo.mark_static_address(value)
        stack.messages = torch.compile(stack.messages, mode="reduce-overhead")
        stack.word_nll = torch.compile(stack.word_nll, mode="reduce-overhead")


def seed_samplers(
    config: Config,
    corpus: BilingualCorpus,
    spoken: list[int],
    replicates: list[int],
    role: str,
) -> SeedSamplers:
    return SeedSamplers(
        [
            WindowSampler(
                corpus.parts[TRAIN],
                spoken,
                config.context_words,
                config.batch_windows,
                seed_for(config.root_seed, r, role),
            )
            for r in replicates
        ]
    )


def due(step: int, interval: int) -> bool:
    return step % interval == 0


class Session:
    def __init__(
        self,
        config: Config,
        name: str,
        steps: int,
        stack: Stack,
        gauge: LanguageGauge,
        releases: list[int | None],
        device: torch.device,
    ) -> None:
        arm = config.arms[name]
        seeds = len(releases)
        self.config, self.name, self.steps = config, name, steps
        self.stack, self.gauge, self.device = stack, gauge, device
        self.seeds = seeds
        self.optimiser, self.gates = optimisers_for(
            stack,
            config.learning_rate,
            config.gate_rate,
            config.optimiser,
            seeds,
            device,
        )
        self.warmup = math.floor(steps * config.optimiser.warmup_fraction)
        self.holds = [
            Hold(arm.hold, config.hold.fall, arm.ruled, fixed) for fixed in releases
        ]
        marked = config.checkpoints.fractions if name in config.checkpoints.arms else ()
        self.marks = {max(1, round(f * steps)): f for f in marked}
        self.saved: list[dict[float, GraphoneModel]] = [{} for _ in range(seeds)]
        self.history = GateHistory(seeds, arm.mix != EQUAL)
        self.accumulator = Accumulator(seeds, gauge.classes, device)
        self.step_log = StepLog(steps, seeds, gauge.classes, device)
        self.curves: list[list[CurvePoint]] = [[] for _ in range(seeds)]
        self.gauges: list[list[GaugePoint]] = [[] for _ in range(seeds)]
        self.learning = learning_mask(self.holds, device)
        self.started = 0.0

    def log(self, at: int) -> None:
        points = self.accumulator.points(at, time.perf_counter() - self.started)
        for curve, point in zip(self.curves, points, strict=True):
            curve.append(point)
        self.accumulator.reset()
        mean = float(np.mean([point.loss for point in points]))
        label = f"{self.name}/{self.seeds}"
        print(f"{label} step={at}/{self.steps} loss={mean:.4f}", flush=True)

    def measure(self, at: int) -> None:
        for k, point in enumerate(self.gauge.read(self.stack, at)):
            self.gauges[k].append(point)
            if self.holds[k].observe(point.language, point.noise, at):
                self.stack.release(k)
        self.learning = learning_mask(self.holds, self.device)

    def train_on(self, index: int, rows: Rows, expander: Expander) -> None:
        set_rate(
            self.optimiser,
            self.gates,
            self.config.learning_rate,
            index,
            self.steps,
            self.warmup,
        )
        batch = expander(rows).by_seed(self.seeds)
        loss, words = step(
            self.stack,
            self.optimiser,
            self.gates,
            self.learning,
            batch,
            self.config.precision,
            self.config.optimiser.grad_clip,
            self.device,
        )
        at = index + 1
        self.release_fixed(at)
        if at in self.marks:
            self.save(at)
        if self.history.mixed and due(at, self.config.snapshot_interval):
            self.history.take(self.stack, at)
        self.accumulator.add(loss, words, batch)
        self.step_log.add(index, loss, words, batch)
        if due(at, self.config.log_interval) or at == self.steps:
            self.log(at)
        if index == 0 or due(at, self.config.gauge.interval):
            self.measure(at)

    def release_fixed(self, at: int) -> None:
        reached = [k for k, hold in enumerate(self.holds) if hold.reach(at)]
        for k in reached:
            self.stack.release(k)
        if reached:
            self.learning = learning_mask(self.holds, self.device)

    def save(self, at: int) -> None:
        for k, saved in enumerate(self.saved):
            saved[self.marks[at]] = copy.deepcopy(self.stack.model(k)).cpu()

    def run(self, batches: Iterator[Rows], expander: Expander) -> float:
        self.learning = learning_mask(self.holds, self.device)
        if self.history.mixed:
            self.history.take(self.stack, 0)
        self.started = time.perf_counter()
        for index, rows in enumerate(batches):
            self.train_on(index, rows, expander)
        return time.perf_counter() - self.started

    def results(self, replicates: list[int], seconds: float) -> list[Trained]:
        models = [copy.deepcopy(self.stack.model(k)) for k in range(len(replicates))]
        receiver = sum(p.numel() for p in self.stack.template.receiver.parameters())
        return [
            Trained(
                models[k],
                Metrics(
                    arm=self.name,
                    replicate=replicate,
                    steps=self.steps,
                    seeds_together=len(replicates),
                    seconds=seconds,
                    device=device_name(self.device),
                    receiver_parameters=receiver,
                    released_at=self.holds[k].released_at,
                    curve=self.curves[k],
                    gauge=self.gauges[k],
                ),
                self.history.record(k, models[k]),
                self.step_log.record(k),
                self.saved[k],
            )
            for k, replicate in enumerate(replicates)
        ]


@dataclass(frozen=True)
class Prepared:
    session: Session
    expander: Expander
    data: SeedSamplers
    replicates: list[int]


def prepared(
    config: Config,
    sizing: Sizing,
    corpus: BilingualCorpus,
    tables: Tables,
    name: str,
    replicates: list[int],
    releases: list[int | None],
    device: torch.device,
) -> Prepared:
    languages = config.data.languages
    if corpus.languages != languages:
        raise ValueError(f"the corpus holds {corpus.languages}, not {languages}")
    seeds = len(replicates)
    set_seed(seed_for(config.root_seed, replicates[0], "training"))
    stack = Stack(
        [assemble(config, sizing, tables, name, r).to(device) for r in replicates]
    )
    compile_for(stack, device)
    expander = Expander(corpus, tables.codes, tables.lengths, device)
    spoken = corpus.language_ids(languages)
    lessons = seed_samplers(config, corpus, spoken, replicates, "lesson")
    gauge = LanguageGauge(
        [expander(lessons.draw()).by_seed(seeds) for _ in range(config.gauge.batches)],
        len(languages),
        config.precision,
        device,
    )
    session = Session(config, name, sizing.steps, stack, gauge, releases, device)
    data = seed_samplers(config, corpus, spoken, replicates, "data")
    return Prepared(session, expander, data, replicates)


def run(ready: Prepared) -> list[Trained]:
    session = ready.session
    pin = on_cuda(session.device)
    with contextlib.closing(prefetched(ready.data, session.steps, pin)) as batches:
        seconds = session.run(batches, ready.expander)
    return session.results(ready.replicates, seconds)


def train(
    config: Config,
    sizing: Sizing,
    corpus: BilingualCorpus,
    tables: Tables,
    name: str,
    replicates: list[int],
    releases: list[int | None],
    device: torch.device,
) -> list[Trained]:
    return run(
        prepared(config, sizing, corpus, tables, name, replicates, releases, device)
    )
