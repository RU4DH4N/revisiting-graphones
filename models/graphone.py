from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn

from models.shared.numerics import normalise
from models.shared.symbols import BLANK

AVERAGE = "avg"
RULE = "rule"
GLOBAL = "global"
TABLE_NAMES = (AVERAGE, RULE, GLOBAL)

# H = H(sound | letter), G = H(letter | sound), S = silent letter
FEATURES = "HGS"
LETTERS = "letters"
SPLIT = "split"
TRIPLE = "triple"
MIXED_STREAMS = 3
STREAMS = {LETTERS: 1, SPLIT: 2, TRIPLE: MIXED_STREAMS}
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


@dataclass(frozen=True)
class StreamTables:
    letters: np.ndarray
    sounds: np.ndarray
    joint: np.ndarray
    has: np.ndarray
    features: np.ndarray
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

    def forward(self, units: Tensor) -> Tensor:
        streams = [self.letters[units]]
        if self.spec.streams != LETTERS:
            streams.append(self.sounds[units])
        if self.spec.streams == TRIPLE:
            streams.append(self.joint[units])
        if self.spec.mix != EQUAL:
            weights = self.weights()[units]
            streams = [weights[..., k : k + 1] * rows for k, rows in enumerate(streams)]
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
