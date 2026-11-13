from dataclasses import dataclass
from pathlib import Path

import numpy as np

from models.graphone import MIXED_STREAMS


@dataclass(frozen=True)
class CurvePoint:
    step: int
    seconds: float
    loss: float
    nats_per_symbol: list[float]


@dataclass(frozen=True)
class GaugePoint:
    step: int
    language: float
    noise: float


@dataclass(frozen=True)
class DirectionPoint:
    step: int
    share: float
    noise: float


@dataclass(frozen=True)
class Metrics:
    arm: str
    replicate: int
    steps: int
    seeds_together: int
    seconds: float
    device: str
    receiver_parameters: int
    released_at: int | None
    curve: list[CurvePoint]
    gauge: list[GaugePoint]
    direction: list[DirectionPoint]


def read_metrics(raw: dict) -> Metrics:
    return Metrics(
        **{
            **raw,
            "curve": [CurvePoint(**point) for point in raw["curve"]],
            "gauge": [GaugePoint(**point) for point in raw["gauge"]],
            "direction": [
                DirectionPoint(**point) for point in raw.get("direction", [])
            ],
        }
    )


@dataclass(frozen=True)
class GateRecord:
    final: np.ndarray
    snapshot_steps: np.ndarray
    snapshots: np.ndarray


def no_gates() -> GateRecord:
    return GateRecord(
        np.zeros((0, MIXED_STREAMS)),
        np.zeros(0, dtype=np.int64),
        np.zeros((0, 0, MIXED_STREAMS)),
    )


@dataclass(frozen=True)
class StepRecord:
    loss: np.ndarray
    nats: np.ndarray
    symbols: np.ndarray


def read_steps(path: Path) -> StepRecord:
    found = np.load(path)
    return StepRecord(found["loss"], found["nats"], found["symbols"])


def save_steps(path: Path, record: StepRecord) -> None:
    np.savez_compressed(
        path, loss=record.loss, nats=record.nats, symbols=record.symbols
    )


def read_gates(path: Path) -> GateRecord:
    found = np.load(path)
    return GateRecord(found["final"], found["snapshot_steps"], found["snapshots"])


def save_gates(path: Path, record: GateRecord) -> None:
    np.savez_compressed(
        path,
        final=record.final,
        snapshot_steps=record.snapshot_steps,
        snapshots=record.snapshots,
    )


@dataclass(frozen=True)
class EvaluationRecord:
    bits_per_byte: float
    minimal_pairs: float
    pos_accuracy: float
    pos_selectivity: float
    pos_transfer: float
    retrieval: float
    retrieval_csls: float
    cka: float
    by_language: dict[str, float]
    words: int


BITS_PER_BYTE = "bits_per_byte"
MINIMAL_PAIRS = "minimal_pairs"
POS_ACCURACY = "pos_accuracy"
POS_SELECTIVITY = "pos_selectivity"
POS_TRANSFER = "pos_transfer"
RETRIEVAL = "retrieval"
RETRIEVAL_CSLS = "retrieval_csls"
CKA = "cka"
LANGUAGE_LATE = "language_late"
SCORES = (
    BITS_PER_BYTE,
    MINIMAL_PAIRS,
    POS_ACCURACY,
    POS_SELECTIVITY,
    POS_TRANSFER,
    RETRIEVAL,
    RETRIEVAL_CSLS,
    CKA,
)


def read_evaluation(raw: dict) -> EvaluationRecord:
    return EvaluationRecord(**raw)
