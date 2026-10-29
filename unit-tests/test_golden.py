import hashlib

import numpy as np

from data.graphones import has_sound, joint_rows, stream_rows
from framework.seeds import seed_for
from models.graphone import GraphoneCode, MixSpec, StreamTables, position_codes
from models.shared.symbols import BLANK

ROOT_SEED = 20261005
KEYS = ["a/a", "a/ə", "b/b", "b", "t/", "", "é/e"]


def digest(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()[:16]


def test_seeds_are_pinned() -> None:
    assert seed_for(20261005, 0, "position-codes") == 326964972
    assert seed_for(20261005, 3, "init") == 2006210802
    assert seed_for(20261005, 1, "data") == 801286733


def test_position_codes_are_pinned() -> None:
    assert position_codes(2, 8, 11).tolist() == [
        [-1.0, -1.0, 1.0, -1.0, 1.0, 1.0, 1.0, -1.0],
        [-1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0],
    ]
    codebook = seed_for(20261005, 0, "position-codes")
    assert digest(position_codes(24, 140, codebook)) == "9b2e05c0dcfccece"


def test_stream_codes_are_pinned() -> None:
    letters, sounds = stream_rows(KEYS, 2, 64, seed_for(ROOT_SEED, "stream-codes"))
    has = np.concatenate([has_sound(KEYS)] * 2)
    assert digest(letters) == "a83b395271e59b43"
    assert digest(sounds) == "4cc88d0a1aff549f"
    assert digest(has) == "aaef829b62e804f1"
    joint = joint_rows(KEYS, 2, 64, seed_for(ROOT_SEED, "joint-codes"))
    assert digest(joint) == "d2f68ce1dc30be53"


def test_the_gate_starts_from_the_same_weights_whatever_came_before() -> None:
    rows = BLANK + 1
    ones = np.ones((rows, 8), dtype=np.int8)
    tables = StreamTables(
        ones, ones, ones, np.ones(rows, bool), np.zeros((rows, 3), np.float32), None, 4
    )
    spec = MixSpec(
        "triple",
        "learned",
        "HGS",
        False,
        False,
        64,
        seed_for(ROOT_SEED, 0, "gate"),
        seed_for(ROOT_SEED, 0, "shuffle"),
    )
    code = GraphoneCode(spec, tables, 11)
    assert digest(code.gate_hidden.weight.detach().numpy()) == "91872c395d2cce16"
