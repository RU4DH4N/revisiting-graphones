from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn

from models.shared.numerics import NORM_FLOOR, normalise
from models.shared.symbols import BLANK

AVERAGE = "avg"
RULE = "rule"
GLOBAL = "global"
TABLE_NAMES = (AVERAGE, RULE, GLOBAL)

# H = H(sound | letter), G = H(letter | sound), S = silent letter
FEATURES = "HGS"
LETTERS = "letters"
BYTES = "bytes"
SPLIT = "split"
TRIPLE = "triple"
MIXED_STREAMS = 3
STREAMS = {LETTERS: 1, BYTES: 1, SPLIT: 2, TRIPLE: MIXED_STREAMS}
EQUAL = "equal"
LEARNED = "learned"
KEPT = "kept"
TABLE = "table"
MIXES = (EQUAL, LEARNED, KEPT, TABLE)
GATED = (LEARNED, KEPT)
# units with no sound have empty sound and joint streams, so the letter takes all
LETTERS_ONLY = (1.0, 0.0, 0.0)
POSITIONS = "positions"
DIRECTION_EPS = 1e-12
RELEASED = "released"
SLOT_VARIABLES = 2


def position_codes(slots: int, dims: int, seed: int) -> NDArray[np.float32]:
    signs = np.array([-1.0, 1.0], dtype=np.float32)
    return np.random.default_rng(seed).choice(signs, size=(slots, dims))


@dataclass(frozen=True)
class MixSpec:
    streams: str
    mix: str
    features: str
    shuffled: bool
    switch: bool
    gate_hidden: int
    gate_seed: int
    shuffle_seed: int
    rotation: bool
    rotation_seed: int


@dataclass(frozen=True)
class StreamTables:
    letters: np.ndarray
    sounds: np.ndarray
    joint: np.ndarray
    has: np.ndarray
    features: np.ndarray
    uses: np.ndarray
    table: np.ndarray | None
    slots: int


def gate_features(
    features: np.ndarray, has: np.ndarray, kept: str, shuffled: bool, seed: int
) -> np.ndarray:
    chosen = features.copy()
    chosen[:, [name not in kept for name in FEATURES]] = 0.0
    if shuffled:
        rows = np.flatnonzero(has)
        chosen[rows] = chosen[rows[np.random.default_rng(seed).permutation(len(rows))]]
    return chosen


def distinct_terms(
    units: Tensor, slots: int
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    flat = units.reshape(-1, units.shape[-1])
    live = flat != BLANK
    length = live.sum(-1)
    start = torch.arange(flat.shape[-1], device=units.device).expand_as(flat)
    end = live.flip(-1).cumsum(-1).flip(-1) - live.long()
    keys = ((flat * slots + start) * slots + end)[live]
    found, inverse = torch.unique(keys, return_inverse=True)
    offsets = torch.cumsum(length, 0) - length
    return (
        found // (slots * slots),
        found // slots % slots,
        found % slots,
        inverse,
        offsets,
    )


def stream(values: np.ndarray, kept: np.ndarray) -> Tensor:
    return torch.from_numpy(values.astype(np.float32) * kept[:, None])


class GraphoneCode(nn.Module):
    PER_SEED = (POSITIONS, RELEASED)
    letters: Tensor
    sounds: Tensor
    joint: Tensor
    has: Tensor
    positions: Tensor
    features: Tensor
    table: Tensor
    letters_only: Tensor
    released: Tensor
    gate_hidden: nn.Linear
    gate_out: nn.Linear
    unit_variables: Tensor
    gate_rotate_hidden: nn.Linear
    gate_rotate_angles: nn.Linear
    gate_mirror_out: nn.Linear

    def __init__(self, spec: MixSpec, tables: StreamTables, seed: int) -> None:
        super().__init__()
        self.spec = spec
        self.mixed = STREAMS[spec.streams]
        self.dims = self.mixed * tables.letters.shape[1]
        has = tables.has.astype(bool)
        has[BLANK] = False
        live = np.ones(len(has), dtype=np.float32)
        live[BLANK] = 0.0
        self.register_buffer("letters", stream(tables.letters, live), persistent=False)
        self.register_buffer("sounds", stream(tables.sounds, has), persistent=False)
        self.register_buffer("has", torch.from_numpy(has), persistent=False)
        self.register_buffer(
            POSITIONS,
            torch.from_numpy(position_codes(tables.slots, self.dims, seed)),
            persistent=False,
        )
        if spec.streams == TRIPLE:
            self.register_buffer("joint", stream(tables.joint, has), persistent=False)
        if spec.mix in GATED:
            self.build_gate(spec, tables)
        if spec.rotation:
            self.build_rotation(spec, tables)
        if spec.mix == TABLE:
            if tables.table is None:
                raise ValueError("a table mix needs a table")
            self.register_buffer(
                "table",
                torch.from_numpy(tables.table.astype(np.float32)),
                persistent=False,
            )
        self.register_buffer(
            "letters_only",
            torch.tensor(LETTERS_ONLY[: self.mixed]),
            persistent=False,
        )
        self.register_buffer(RELEASED, torch.tensor(not spec.switch))

    def build_gate(self, spec: MixSpec, tables: StreamTables) -> None:
        features = gate_features(
            tables.features, tables.has, spec.features, spec.shuffled, spec.shuffle_seed
        )
        self.register_buffer(
            "features", torch.from_numpy(features.astype(np.float32)), persistent=False
        )
        state = torch.get_rng_state()
        torch.manual_seed(spec.gate_seed)
        self.gate_hidden = nn.Linear(len(FEATURES), spec.gate_hidden)
        learned = self.mixed if spec.mix == LEARNED else self.mixed - 1
        self.gate_out = nn.Linear(spec.gate_hidden, learned)
        nn.init.zeros_(self.gate_out.weight)
        nn.init.ones_(self.gate_out.bias)
        torch.set_rng_state(state)

    def build_rotation(self, spec: MixSpec, tables: StreamTables) -> None:
        if spec.mix not in GATED:
            raise ValueError("a rotation reads the gate's features")
        if self.dims % 2:
            raise ValueError("a rotation turns coordinate pairs")
        uses = np.log1p(tables.uses.astype(np.float64))
        scaled = uses / max(float(uses.max()), 1.0)
        unit = np.stack([~self.has.numpy(), scaled], axis=-1).astype(np.float32)
        self.register_buffer(
            "unit_variables",
            torch.cat([self.features, torch.from_numpy(unit)], dim=-1),
            persistent=False,
        )
        state = torch.get_rng_state()
        torch.manual_seed(spec.rotation_seed)
        self.gate_rotate_hidden = nn.Linear(
            self.unit_variables.shape[1] + SLOT_VARIABLES, spec.gate_hidden
        )
        self.gate_rotate_angles = nn.Linear(spec.gate_hidden, self.dims // 2)
        self.gate_mirror_out = nn.Linear(spec.gate_hidden, self.dims)
        nn.init.zeros_(self.gate_rotate_angles.weight)
        nn.init.zeros_(self.gate_rotate_angles.bias)
        nn.init.zeros_(self.gate_mirror_out.weight)
        with torch.no_grad():
            self.gate_mirror_out.bias.copy_(torch.randn(self.dims))
        torch.set_rng_state(state)

    def rotation_hidden(self, unit: Tensor, start: Tensor, end: Tensor) -> Tensor:
        slots = self.positions.shape[0]
        inputs = torch.cat(
            [
                self.unit_variables[unit],
                (start.float() / slots)[:, None],
                (end.float() / slots)[:, None],
            ],
            dim=-1,
        )
        return nn.functional.gelu(self.gate_rotate_hidden(inputs))

    def rotated_words(self, units: Tensor) -> Tensor:
        unit, start, end, inverse, offsets = distinct_terms(
            units, self.positions.shape[0]
        )
        terms = torch.cat(self.weighted(unit), dim=-1) * self.positions[start]
        table = self.turned(terms, self.rotation_hidden(unit, start, end))
        words = nn.functional.embedding_bag(inverse, table, offsets, mode="sum")
        return words.reshape(*units.shape[:-1], self.dims)

    def turned(self, terms: Tensor, hidden: Tensor) -> Tensor:
        angles = self.gate_rotate_angles(hidden).to(terms.dtype)
        mirror = self.gate_mirror_out(hidden).to(terms.dtype)
        pairs = terms.unflatten(-1, (self.dims // 2, 2))
        first, second = pairs[..., 0], pairs[..., 1]
        cos, sin = torch.cos(angles), torch.sin(angles)
        rotated = torch.stack(
            [first * cos - second * sin, first * sin + second * cos], dim=-1
        ).flatten(-2)
        norm = (mirror * mirror).sum(-1, keepdim=True) + NORM_FLOOR
        along = (mirror * rotated).sum(-1, keepdim=True) / norm
        return rotated - 2 * mirror * along

    def release(self) -> None:
        self.released.fill_(True)

    def directions(self) -> Tensor:
        if self.spec.mix in GATED:
            raw = self.gate_out(nn.functional.gelu(self.gate_hidden(self.features)))
            unit = normalise(raw, DIRECTION_EPS)
            if self.spec.mix == LEARNED:
                return unit
            kept = torch.full((len(raw), 1), self.mixed**-0.5, device=raw.device)
            return torch.cat([kept, unit * (1 - 1 / self.mixed) ** 0.5], dim=-1)
        equal = torch.full(
            (len(self.has), self.mixed), self.mixed**-0.5, device=self.has.device
        )
        if self.spec.mix == EQUAL:
            return equal
        return torch.where(self.released, self.table, equal)

    # sqrt(streams) * direction keeps the summed squared weights at the stream count, as in the equal mix
    def weights(self) -> Tensor:
        directions = self.directions().to(self.letters.dtype)
        return torch.where(
            self.has[:, None], self.mixed**0.5 * directions, self.letters_only
        )

    def weighted(self, units: Tensor) -> list[Tensor]:
        streams = [self.letters[units]]
        if self.mixed > 1:
            streams.append(self.sounds[units])
        if self.spec.streams == TRIPLE:
            streams.append(self.joint[units])
        if self.spec.mix == EQUAL:
            return streams
        weights = self.weights()[units]
        return [weights[..., k : k + 1] * rows for k, rows in enumerate(streams)]

    def forward(self, units: Tensor) -> Tensor:
        if self.spec.rotation:
            return self.rotated_words(units)
        streams = self.weighted(units)
        positions = self.positions[: units.shape[-1]].split(
            self.letters.shape[1], dim=-1
        )
        return torch.cat(
            [
                torch.einsum("...sd,sd->...d", rows, codes)
                for rows, codes in zip(streams, positions, strict=True)
            ],
            dim=-1,
        )
