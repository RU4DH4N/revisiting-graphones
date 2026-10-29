import array
import hashlib
import math
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

from data.benchmarks import (
    BLIMP,
    FLORES,
    TREEBANK_PARTS,
    Benchmarks,
    BenchmarkSources,
    treebank,
)
from data.corpus import split_words, to_units
from data.splits import SPLITS, TRAIN
from framework.storage import file_sha256, read_json, write_json

FORMAT = "bilingual-events-v5"
BENCHMARK_PARTS = (FLORES, *TREEBANK_PARTS, BLIMP)
IDS = "ids"
OFFSETS = "offsets"
LANGUAGES = "languages"
METADATA = "metadata.json"
VOCABULARY_BYTES = "vocabulary-bytes.npy"
VOCABULARY_LENGTHS = "vocabulary-lengths.npy"


def part_file(split: str, part: str) -> str:
    return f"{split}-{part}.npy"


def tags_file(part: str) -> str:
    return f"{part}-tags.npy"


CORPUS_FILES = (
    METADATA,
    VOCABULARY_BYTES,
    VOCABULARY_LENGTHS,
    *(
        part_file(split, part)
        for split in (*SPLITS, *BENCHMARK_PARTS)
        for part in (IDS, OFFSETS, LANGUAGES)
    ),
    *(tags_file(part) for part in TREEBANK_PARTS),
)
LENGTH_SAMPLE_STRIDE = 10


def split_of(text: str, shares: dict[str, float]) -> str:
    digest = hashlib.sha256(text.encode()).digest()
    point = int.from_bytes(digest[:8], "big") / 2**64
    edge = 0.0
    for split in SPLITS:
        edge += shares[split]
        if point < edge:
            return split
    return SPLITS[-1]


def word_prefix(raw: bytes) -> str:
    text = raw.decode(errors="ignore")
    cut = max(text.rfind(" "), text.rfind("\n"))
    return text[:cut] if cut > 0 else ""


def collect(texts: Iterable[str], targets: dict[str, int]) -> dict[str, list[str]]:
    total = sum(targets.values())
    shares = {split: targets[split] / total for split in SPLITS}
    found: dict[str, list[str]] = {split: [] for split in SPLITS}
    used: dict[str, int] = dict.fromkeys(SPLITS, 0)
    full: set[str] = set()
    for text in texts:
        split = split_of(text, shares)
        if split in full:
            continue
        raw = text.encode()
        room = targets[split] - used[split]
        if len(raw) >= room:
            text = word_prefix(raw[:room])
            full.add(split)
        if text:
            found[split].append(text)
            used[split] += len(text.encode())
        if len(full) == len(SPLITS):
            return found
    missing = {s: targets[s] - used[s] for s in SPLITS if s not in full}
    raise ValueError(f"the source ran out before the byte targets were met: {missing}")


def unit_bytes(documents: list[str], coverage: float, minimum: int) -> int:
    lengths = sorted(
        len(piece.encode())
        for text in documents[::LENGTH_SAMPLE_STRIDE]
        for piece in split_words(text)
    )
    if not lengths:
        raise ValueError("the training sample has no words")
    index = min(len(lengths) - 1, math.ceil(coverage * len(lengths)) - 1)
    return max(minimum, lengths[index])


class Vocabulary:
    def __init__(self, max_bytes: int) -> None:
        self.max_bytes = max_bytes
        self.index: dict[str, int] = {}

    def encode(self, text: str) -> array.array:
        ids = array.array("i")
        for unit in to_units(text, self.max_bytes):
            ids.append(self.index.setdefault(unit, len(self.index)))
        return ids

    def table(self) -> tuple[np.ndarray, np.ndarray]:
        table = np.zeros((len(self.index), self.max_bytes), dtype=np.uint8)
        lengths = np.zeros(len(self.index), dtype=np.uint8)
        for unit, row in self.index.items():
            raw = unit.encode()
            table[row, : len(raw)] = np.frombuffer(raw, dtype=np.uint8)
            lengths[row] = len(raw)
        return table, lengths


@dataclass(frozen=True)
class Encoded:
    ids: np.ndarray
    offsets: np.ndarray
    languages: np.ndarray

    def word_languages(self) -> np.ndarray:
        return np.repeat(self.languages, np.diff(self.offsets))

    def files(self, split: str) -> dict[str, np.ndarray]:
        return {
            part_file(split, IDS): self.ids,
            part_file(split, OFFSETS): self.offsets,
            part_file(split, LANGUAGES): self.languages,
        }


def load_part(root: Path, split: str) -> Encoded:
    return Encoded(
        np.load(root / part_file(split, IDS), mmap_mode="r"),
        np.load(root / part_file(split, OFFSETS)),
        np.load(root / part_file(split, LANGUAGES)),
    )


def encode_documents(
    vocabulary: Vocabulary,
    groups: list[tuple[int, list[str]]],
) -> Encoded:
    ids = array.array("i")
    offsets = [0]
    languages = []
    for language, documents in groups:
        for text in documents:
            encoded = vocabulary.encode(text)
            if len(encoded) < 2:
                continue
            ids.extend(encoded)
            offsets.append(len(ids))
            languages.append(language)
    return Encoded(
        np.frombuffer(ids, dtype=np.int32).copy(),
        np.array(offsets, dtype=np.int64),
        np.array(languages, dtype=np.int8),
    )


@dataclass(frozen=True)
class DataSettings:
    languages: list[str]
    sources: dict[str, str]
    bytes: dict[str, int]
    unit_coverage: float
    min_unit_bytes: int
    benchmarks: BenchmarkSources


def encode_treebanks(
    vocabulary: Vocabulary, treebanks: dict[str, list[bytes]]
) -> tuple[dict[str, Encoded], dict[str, np.ndarray]]:
    encoded, tags = {}, {}
    for part, raws in treebanks.items():
        sentences = [
            [
                found
                for found in treebank(raw, vocabulary.max_bytes)
                if len(found.tags) >= 2
            ]
            for raw in raws
        ]
        encoded[part] = encode_documents(
            vocabulary,
            [(k, [found.text for found in kept]) for k, kept in enumerate(sentences)],
        )
        tags[part] = np.array(
            [tag for kept in sentences for found in kept for tag in found.tags],
            dtype=np.int8,
        )
        if len(tags[part]) != len(encoded[part].ids):
            raise ValueError(f"{part} tags do not line up with its units")
    return encoded, tags


def encode_pairs(vocabulary: Vocabulary, pairs: list[list[tuple[str, str]]]) -> Encoded:
    kept = [
        [
            pair
            for pair in found
            if all(len(to_units(text, vocabulary.max_bytes)) >= 2 for text in pair)
        ]
        for found in pairs
    ]
    return encode_documents(
        vocabulary,
        [
            (k, [text for pair in chosen for text in pair])
            for k, chosen in enumerate(kept)
        ],
    )


def encode_benchmarks(
    vocabulary: Vocabulary, benchmarks: Benchmarks
) -> tuple[dict[str, Encoded], dict[str, np.ndarray]]:
    flores = encode_documents(vocabulary, list(enumerate(benchmarks.flores)))
    if len(flores.languages) != sum(map(len, benchmarks.flores)):
        raise ValueError("a FLORES sentence encoded to fewer than two units")
    treebanks, tags = encode_treebanks(vocabulary, benchmarks.treebanks)
    blimp = encode_pairs(vocabulary, benchmarks.pairs)
    return {FLORES: flores, **treebanks, BLIMP: blimp}, tags


def statistics(encoded: dict[str, Encoded], languages: list[str]) -> dict:
    return {
        split: {
            "documents": len(value.languages),
            "words": len(value.ids),
            "words_by_language": {
                language: int(np.diff(value.offsets)[value.languages == index].sum())
                for index, language in enumerate(languages)
            },
        }
        for split, value in encoded.items()
    }


def save_arrays(root: Path, arrays: dict[str, np.ndarray]) -> dict[str, str]:
    root.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for name, values in arrays.items():
        path = root / name
        np.save(path, values, allow_pickle=False)
        hashes[path.name] = file_sha256(path)
    return hashes


def build(
    data: DataSettings,
    root: Path,
    sources: Iterator[tuple[str, Iterable[str]]],
    benchmarks: Benchmarks,
) -> None:
    languages = list(data.languages)
    collected = {language: collect(texts, data.bytes) for language, texts in sources}
    if sorted(collected) != sorted(languages):
        raise ValueError("every configured language needs exactly one source")
    training = [text for language in languages for text in collected[language][TRAIN]]
    vocabulary = Vocabulary(
        unit_bytes(training, data.unit_coverage, data.min_unit_bytes)
    )
    encoded = {
        split: encode_documents(
            vocabulary,
            [(k, collected[language][split]) for k, language in enumerate(languages)],
        )
        for split in SPLITS
    }
    tested, tags = encode_benchmarks(vocabulary, benchmarks)
    encoded.update(tested)
    table, lengths = vocabulary.table()
    arrays = {VOCABULARY_BYTES: table, VOCABULARY_LENGTHS: lengths}
    for split, value in encoded.items():
        arrays.update(value.files(split))
    arrays.update({tags_file(part): values for part, values in tags.items()})
    write_json(
        root / METADATA,
        {
            "format": FORMAT,
            "languages": languages,
            "max_unit_bytes": vocabulary.max_bytes,
            "vocabulary": len(lengths),
            "statistics": statistics(encoded, languages),
            "sources": asdict(data),
            "sha256": save_arrays(root, arrays),
        },
    )


# https://huggingface.co/datasets/HuggingFaceFW/fineweb
def parquet_texts(path: str) -> Iterator[str]:

    with HfFileSystem().open(path, "rb") as handle:
        for batch in pq.ParquetFile(handle).iter_batches(columns=["text"]):
            yield from (
                t for t in batch.column(0).to_pylist() if isinstance(t, str) and t
            )


def shards(folder: str) -> list[str]:
    found = HfFileSystem().ls(folder, detail=False)
    return sorted(str(name) for name in found if str(name).endswith(".parquet"))


def source_texts(folder: str) -> Iterator[str]:
    for path in shards(folder):
        yield from parquet_texts(path)


def fineweb_sources(data: DataSettings) -> Iterator[tuple[str, Iterable[str]]]:
    for language in data.languages:
        yield language, source_texts(data.sources[language])


class BilingualCorpus:
    def __init__(self, root: Path) -> None:
        metadata = read_json(root / METADATA)
        if metadata["format"] != FORMAT:
            raise ValueError(f"unknown corpus format in {root}")
        for name, expected in metadata["sha256"].items():
            if file_sha256(root / name) != expected:
                raise ValueError(f"corpus file does not match its hash: {name}")
        self.metadata = metadata
        self.languages: list[str] = metadata["languages"]
        self.max_bytes: int = metadata["max_unit_bytes"]
        self.hashes: dict[str, str] = metadata["sha256"]
        self.table = np.load(root / VOCABULARY_BYTES)
        self.lengths = np.load(root / VOCABULARY_LENGTHS)
        self.parts = {
            split: load_part(root, split) for split in (*SPLITS, *BENCHMARK_PARTS)
        }
        self.tags = {part: np.load(root / tags_file(part)) for part in TREEBANK_PARTS}

    def train_words(self) -> int:
        return self.metadata["statistics"][TRAIN]["words"]

    def language_ids(self, languages: list[str]) -> list[int]:
        unknown = set(languages) - set(self.languages)
        if unknown:
            raise ValueError(f"languages not in the corpus: {sorted(unknown)}")
        return [self.languages.index(language) for language in languages]
