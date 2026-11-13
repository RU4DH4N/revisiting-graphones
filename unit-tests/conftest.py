import json
import random
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import pytest

import data.bilingual as bilingual
from data.benchmarks import Benchmarks
from data.bilingual import BilingualCorpus
from data.graphones import (
    PHONE_MARK,
    Tables,
    align_corpus,
    build_units,
    read_tables,
    type_counts,
    word_parts,
    write_tables,
)
from data.splits import TRAIN
from models.bilingual import GraphoneModel
from models.graphone import EQUAL, LEARNED, SPLIT, STREAMS, TABLE, TRIPLE
from workers import design
from workers.arms import assemble
from workers.design import Sizing
from workers.settings import Config, parse

EN = "the cat sat on a mat while dogs ran over green hills and people talked about weather".split()
FR = "le chat était assis sur un tapis pendant que des chiens couraient très vite à côté".split()
LANGUAGES = ["en", "fr"]
TRUNK_WIDTH = 48
VOICES = {"en": "en-test", "fr": "fr-test"}


def synthetic(words: Sequence[str], seed: int, documents: int) -> Iterator[str]:
    rng = random.Random(seed)
    for _ in range(documents):
        yield " ".join(rng.choice(words) for _ in range(rng.randint(30, 60)))


def tag_of(word: str) -> str:
    return ("NOUN", "VERB", "DET", "ADP")[sum(map(ord, word)) % 4]


def conllu(words: Sequence[str], seed: int, sentences: int) -> bytes:
    rng = random.Random(seed)
    blocks = []
    for _ in range(sentences):
        chosen = [rng.choice(words) for _ in range(rng.randint(5, 9))]
        lines = [
            f"{i}\t{word}\t{word}\t{tag_of(word)}\t_\t_\t0\tdep\t_\t_"
            for i, word in enumerate(chosen, start=1)
        ]
        lines.append(f"{len(chosen) + 1}\t.\t.\tPUNCT\t_\t_\t0\tpunct\t_\t_")
        lines[-2] = lines[-2].removesuffix("_") + "SpaceAfter=No"
        blocks.append("\n".join(lines))
    return ("\n\n".join(blocks) + "\n\n").encode()


def benchmarks() -> Benchmarks:
    rng = random.Random(9)
    flores = [
        [" ".join(rng.choice(words) for _ in range(8)) for _ in range(24)]
        for words in (EN, FR)
    ]
    pairs = []
    for words in (EN, FR):
        found = []
        for _ in range(20):
            good = [rng.choice(words) for _ in range(6)]
            found.append((" ".join(good), " ".join(reversed(good))))
        pairs.append(found)
    return Benchmarks(
        flores=flores,
        treebanks={
            "ud-dev": [conllu(EN, 4, 40), conllu(FR, 5, 40)],
            "ud-test": [conllu(EN, 6, 40), conllu(FR, 7, 40)],
        },
        pairs=pairs,
    )


def sounded(word: str) -> str:
    found = []
    for i, ch in enumerate(word):
        if ch == "c":
            found.append("s" if word[i + 1 : i + 2] in ("e", "i") else "k")
        elif not (ch == "e" and i == len(word) - 1 and len(word) > 2):
            found.append(ch)
    return PHONE_MARK.join(found)


def spelled(voice: str, words: list[str]) -> list[str]:
    if voice not in VOICES.values():
        raise ValueError(f"unknown voice {voice}")
    return [sounded(word) for word in words]


def arm(code_dims: int, streams: str, mix: str, **changes: object) -> dict:
    return {
        "seeds": 1,
        "projection": STREAMS[streams] * code_dims != TRUNK_WIDTH,
        "input_mlp": False,
        "code_dims": code_dims,
        "matched": False,
        "receiver_scale": 1,
        "streams": streams,
        "mix": mix,
        "features": "HGS",
        "shuffled": False,
        "hold": False,
        "release_fraction": None,
        "release_from": None,
        "table": None,
        "rotation": False,
        "trigger": "language",
        "projection_rank": None,
        **changes,
    }


ARMS = {
    "baseline": arm(32, SPLIT, EQUAL),
    "baseline-mlp": arm(32, SPLIT, EQUAL, input_mlp=True),
    "triple-direct": arm(16, TRIPLE, EQUAL),
    "learned-direct": arm(16, TRIPLE, LEARNED, hold=True),
    "learned-direct-matched": arm(16, TRIPLE, LEARNED, hold=True, matched=True),
    "fixed-at": arm(16, TRIPLE, LEARNED, hold=True, release_from="learned-direct"),
    "table-avg": arm(16, TRIPLE, TABLE, table="avg"),
}
DOWNLOAD = {"url": "https://example.invalid/file", "sha256": "0" * 64}


def data_section() -> dict:
    return {
        "languages": LANGUAGES,
        "sources": {"en": "synthetic", "fr": "synthetic"},
        "bytes": {"train": 6000, "selection": 1500, "reporting": 1500},
        "unit_coverage": 0.999,
        "min_unit_bytes": 4,
        "benchmarks": {
            "flores": {**DOWNLOAD, "members": {"en": "en", "fr": "fr"}},
            "treebanks": {"en": {"dev": DOWNLOAD}, "fr": {"dev": DOWNLOAD}},
            "minimal_pairs": {"en": DOWNLOAD, "fr": DOWNLOAD},
        },
    }


def graphone_section() -> dict:
    return {
        "sample": 200,
        "rounds": 2,
        "costs": {
            "silent": 1.5,
            "double": 1.5,
            "extra": 3.0,
            "pair": 4.5,
            "unseen": 6.0,
        },
        "voices": VOICES,
    }


def model_section() -> dict:
    return {
        "head_dimensions": 64,
        "width_per_layer": 64,
        "mlp_ratio": 4.0,
        "multiple_of": 64,
        "norm_eps": 1e-5,
        "rope_theta": 10_000.0,
        "initialiser_range": 0.02,
        "receiver_fraction": 0.125,
        "message_rank_divisor": 4,
        "gate_hidden": 64,
    }


def raw_config(arms: dict) -> dict:
    return {
        "data": data_section(),
        "graphones": graphone_section(),
        "trunk_parameters": 60000,
        "trunk_width": TRUNK_WIDTH,
        "context_words": 6,
        "batch_windows": 4,
        "passes": 0.2,
        "model": model_section(),
        "root_seed": 3,
        "learning_rate": 0.003,
        "gate_rate": 20.0,
        "optimiser": {
            "weight_decay": 0.1,
            "grad_clip": 1.0,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "warmup_fraction": 0.01,
        },
        "hold": {"fall": 0.1},
        "precision": "fp32",
        "log_interval": 5,
        "snapshot_interval": 250,
        "gauge": {"interval": 5, "batches": 2},
        "checkpoints": {
            "fractions": [0.5],
            "arms": [name for name in ("learned-direct",) if name in arms],
        },
        "tables": {"source": "learned-direct", "floor": 0.01},
        "evaluation": {"probe_steps": 100, "probe_decay": 1e-4, "neighbours": 10},
        "analysis": {"late": 0.5},
        "arms": arms,
    }


def config() -> dict:
    return raw_config(ARMS)


def set_value(path: tuple[str, ...], value: object, settings: dict) -> None:
    section = settings
    for key in path[:-1]:
        section = section[key]
    section[path[-1]] = value


def model_with(tiny: "Tiny", settings: Config, name: str) -> GraphoneModel:
    return assemble(settings, tiny.sizing, tiny.tables, name, 0)


def model_for(tiny: "Tiny", name: str) -> GraphoneModel:
    return model_with(tiny, tiny.settings, name)


def with_arm(settings: Config, name: str, **changes: object) -> Config:
    changed = replace(settings.arms[name], **changes)
    return replace(settings, arms={**settings.arms, name: changed})


@dataclass(frozen=True)
class Tiny:
    root: Path
    raw: dict
    settings: Config
    corpus: BilingualCorpus
    tables: Tables
    sizing: Sizing


def built(root: Path, raw: dict) -> Tiny:
    settings = parse(raw)
    sources = iter([("en", synthetic(EN, 1, 400)), ("fr", synthetic(FR, 2, 400))])
    bilingual.build(settings.data, root / "corpus", sources, benchmarks())
    corpus = BilingualCorpus(root / "corpus")
    parts = word_parts(corpus)
    counts = type_counts(corpus)
    alignments = align_corpus(corpus, parts, counts, settings.graphones, spelled)
    write_tables(
        root / "graphones",
        build_units(parts, alignments, corpus.max_bytes),
        counts,
        settings.code_widths(),
        settings.root_seed,
    )
    words = corpus.metadata["statistics"][TRAIN]["words"]
    found = design.sized(
        settings, corpus.max_bytes, words, design.steps(settings, words)
    )
    (root / "config.json").write_text(json.dumps(raw))
    return Tiny(root, raw, settings, corpus, read_tables(root / "graphones"), found)


@pytest.fixture(scope="session")
def tiny(tmp_path_factory: pytest.TempPathFactory) -> Tiny:
    return built(tmp_path_factory.mktemp("tiny"), config())
