import contextlib
import queue
import threading
from collections.abc import Generator, Iterator
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from data.bilingual import BilingualCorpus, Encoded
from framework.runtime import on_cuda
from models.bilingual import IGNORE, PAD, START, STOP

PREFETCH = 4


@dataclass(frozen=True)
class Rows:
    words: Tensor
    languages: Tensor
    scored: Tensor
    documents: Tensor


@dataclass(frozen=True)
class WordBatch:
    context: Tensor
    inputs: Tensor
    targets: Tensor
    languages: Tensor
    scored: Tensor
    spelled: Tensor
    documents: Tensor
    types: Tensor

    def by_seed(self, seeds: int) -> "WordBatch":
        return WordBatch(
            context=per_seed(self.context, seeds),
            inputs=per_seed(self.inputs, seeds),
            targets=per_seed(self.targets, seeds),
            languages=per_seed(self.languages, seeds),
            scored=per_seed(self.scored, seeds),
            spelled=per_seed(self.spelled, seeds),
            documents=per_seed(self.documents, seeds),
            types=per_seed(self.types, seeds),
        )


def per_seed(values: Tensor, seeds: int) -> Tensor:
    return values.view(seeds, -1, *values.shape[1:])


class Expander:
    def __init__(
        self,
        corpus: BilingualCorpus,
        codes: np.ndarray,
        sizes: np.ndarray,
        device: torch.device,
    ) -> None:
        self.codes = torch.from_numpy(codes.astype(np.int32, copy=False)).to(device)
        self.sizes = torch.from_numpy(sizes.astype(np.int64)).to(device)
        self.slots = torch.arange(codes.shape[2], device=device)
        self.table = torch.from_numpy(corpus.table).to(device)
        self.lengths = torch.from_numpy(corpus.lengths.astype(np.int64)).to(device)
        self.device = device
        self.overlap = on_cuda(device)
        self.positions = torch.arange(corpus.max_bytes + 1, device=device)

    def __call__(self, rows: Rows) -> WordBatch:
        words = rows.words.to(self.device, non_blocking=self.overlap)
        events = words.shape[1] - 1
        context_ids = words[:, :events]
        languages = rows.languages.to(self.device, non_blocking=self.overlap)
        spoken = languages[:, None].expand_as(context_ids)
        context = (
            self.codes[spoken, context_ids]
            .long()
            .masked_fill(self.slots >= self.sizes[spoken, context_ids][..., None], PAD)
        )
        target_ids = words[:, 1:].reshape(-1)
        letters = self.table[target_ids].long()
        length = self.lengths[target_ids][:, None]
        padded = torch.cat((letters, letters.new_zeros(letters.shape[0], 1)), dim=1)
        targets = torch.where(
            self.positions < length,
            padded,
            torch.where(self.positions == length, STOP, IGNORE),
        )
        shifted = torch.cat(
            (torch.full_like(padded[:, :1], START), padded[:, :-1]), dim=1
        )
        inputs = torch.where(self.positions <= length, shifted, 0)
        scored = rows.scored.to(self.device, non_blocking=self.overlap).reshape(-1)
        return WordBatch(
            context=context,
            inputs=inputs,
            targets=torch.where(scored[:, None], targets, IGNORE),
            languages=languages[:, None].expand(-1, events).reshape(-1),
            scored=scored,
            spelled=torch.where(scored, length[:, 0] + 1, 0),
            documents=rows.documents.to(self.device, non_blocking=self.overlap)[:, None]
            .expand(-1, events)
            .reshape(-1),
            types=target_ids,
        )


class WindowSampler:
    def __init__(
        self, part: Encoded, languages: list[int], events: int, batch: int, seed: int
    ) -> None:
        lengths = np.diff(part.offsets)
        eligible = np.isin(part.languages, languages) & (lengths > events)
        if not eligible.any():
            raise ValueError(f"no document has more than {events} words")
        self.part = part
        self.groups = []
        for language in languages:
            chosen = np.flatnonzero(eligible & (part.languages == language))
            self.groups.append((chosen, (lengths[chosen] - events).cumsum()))
        sizes = np.array(
            [float(c[-1]) if len(c) else 0.0 for _, c in self.groups], dtype=np.float64
        )
        self.shares = sizes / sizes.sum()
        self.events = events
        self.batch = batch
        self.rng = np.random.default_rng(seed)

    def positions(self) -> tuple[np.ndarray, np.ndarray]:
        counts = self.rng.multinomial(self.batch, self.shares)
        found = [
            self.pick(chosen, cumulative, int(count))
            for (chosen, cumulative), count in zip(self.groups, counts, strict=True)
            if count
        ]
        return (
            np.concatenate([documents for documents, _ in found]),
            np.concatenate([offsets for _, offsets in found]),
        )

    def pick(
        self, documents: np.ndarray, cumulative: np.ndarray, count: int
    ) -> tuple[np.ndarray, np.ndarray]:
        draws = self.rng.integers(0, int(cumulative[-1]), size=count)
        slot = np.searchsorted(cumulative, draws, side="right")
        before = np.where(slot == 0, 0, cumulative[np.maximum(slot - 1, 0)])
        return documents[slot], draws - before

    def draw(self) -> Rows:
        documents, within = self.positions()
        starts = self.part.offsets[documents] + within
        words = np.asarray(self.part.ids[starts[:, None] + np.arange(self.events + 1)])
        return Rows(
            words=torch.from_numpy(words.astype(np.int64)),
            languages=torch.from_numpy(self.part.languages[documents].astype(np.int64)),
            scored=torch.ones(self.batch, self.events, dtype=torch.bool),
            documents=torch.from_numpy(documents.astype(np.int64)),
        )


class SeedSamplers:
    def __init__(self, samplers: list[WindowSampler]) -> None:
        self.samplers = samplers

    def draw(self) -> Rows:
        drawn = [sampler.draw() for sampler in self.samplers]
        return Rows(
            words=torch.cat([rows.words for rows in drawn]),
            languages=torch.cat([rows.languages for rows in drawn]),
            scored=torch.cat([rows.scored for rows in drawn]),
            documents=torch.cat([rows.documents for rows in drawn]),
        )


def produce(
    sampler: SeedSamplers,
    steps: int,
    pin: bool,
    buffer: queue.Queue[Rows | BaseException],
    stop: threading.Event,
) -> None:
    made = 0
    try:
        while made < steps and not stop.is_set():
            buffer.put(pinned(sampler.draw(), pin))
            made += 1
    except BaseException as error:
        buffer.put(error)


def taken(buffer: queue.Queue[Rows | BaseException]) -> Rows:
    item = buffer.get()
    if isinstance(item, BaseException):
        raise item
    return item


def prefetched(sampler: SeedSamplers, steps: int, pin: bool) -> Generator[Rows]:
    buffer: queue.Queue[Rows | BaseException] = queue.Queue(maxsize=PREFETCH)
    stop = threading.Event()
    worker = threading.Thread(
        target=produce, args=(sampler, steps, pin, buffer, stop), daemon=True
    )
    worker.start()
    try:
        for _ in range(steps):
            yield taken(buffer)
    finally:
        stop.set()
        drain(buffer, worker)


def pinned(rows: Rows, pin: bool) -> Rows:
    return (
        Rows(
            words=rows.words.pin_memory(),
            languages=rows.languages.pin_memory(),
            scored=rows.scored.pin_memory(),
            documents=rows.documents.pin_memory(),
        )
        if pin
        else rows
    )


def drain(buffer: queue.Queue, worker: threading.Thread) -> None:
    while worker.is_alive():
        with contextlib.suppress(queue.Empty):
            buffer.get(timeout=0.1)
    worker.join()


def evaluation_rows(
    part: Encoded, languages: list[int], events: int, batch: int, order_seed: int
) -> Iterator[Rows]:
    windows: list[tuple[int, int, int]] = []
    lengths = np.diff(part.offsets)
    for document in np.flatnonzero(np.isin(part.languages, languages)):
        start, length = int(part.offsets[document]), int(lengths[document])
        for first in range(0, length - 1, events):
            windows.append(
                (int(document), start + first, min(events, length - 1 - first))
            )
    order = np.random.default_rng(order_seed).permutation(len(windows))
    windows = [windows[index] for index in order]
    for begin in range(0, len(windows), batch):
        chunk = windows[begin : begin + batch]
        words = np.zeros((batch, events + 1), dtype=np.int64)
        scored = np.zeros((batch, events), dtype=bool)
        documents = np.zeros(batch, dtype=np.int64)
        for row, (document, start, count) in enumerate(chunk):
            words[row, : count + 1] = part.ids[start : start + count + 1]
            scored[row, :count] = True
            documents[row] = document
        yield Rows(
            words=torch.from_numpy(words),
            languages=torch.from_numpy(part.languages[documents].astype(np.int64)),
            scored=torch.from_numpy(scored),
            documents=torch.from_numpy(documents),
        )
