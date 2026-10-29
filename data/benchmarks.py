import csv
import hashlib
import io
import sys
import tarfile
from dataclasses import dataclass

from huggingface_hub.utils import http_backoff

from data.corpus import to_units
from data.splits import HALVES

UPOS = (  # https://universaldependencies.org/u/pos/
    "ADJ",
    "ADP",
    "ADV",
    "AUX",
    "CCONJ",
    "DET",
    "INTJ",
    "NOUN",
    "NUM",
    "PART",
    "PRON",
    "PROPN",
    "PUNCT",
    "SCONJ",
    "SYM",
    "VERB",
    "X",
)
UNTAGGED = -1
FLORES = "flores"
BLIMP = "blimp"
UD_DEV = "ud-dev"
UD_TEST = "ud-test"
TREEBANK_PARTS = {UD_DEV: "dev", UD_TEST: "test"}


@dataclass(frozen=True)
class Download:
    url: str
    sha256: str


@dataclass(frozen=True)
class Archive:
    url: str
    sha256: str
    members: dict[str, str]


@dataclass(frozen=True)
class BenchmarkSources:
    flores: Archive
    treebanks: dict[str, dict[str, Download]]
    minimal_pairs: dict[str, Download]


@dataclass(frozen=True)
class Tagged:
    text: str
    tags: list[int]


def checked(source: Download | Archive) -> bytes:
    reply = http_backoff("GET", source.url, follow_redirects=True)
    reply.raise_for_status()
    raw = reply.content
    if hashlib.sha256(raw).hexdigest() != source.sha256:
        raise ValueError(f"{source.url} does not match its pinned hash")
    return raw


def half(key: str) -> str:
    return HALVES[hashlib.sha256(key.encode()).digest()[0] % 2]


def flores_sentences(flores: Archive, languages: list[str]) -> list[list[str]]:
    found = []
    with tarfile.open(fileobj=io.BytesIO(checked(flores)), mode="r:gz") as archive:
        for language in languages:
            handle = archive.extractfile(flores.members[language])
            if handle is None:
                raise ValueError(f"the FLORES archive has no file for {language}")
            found.append(handle.read().decode().splitlines())
    if len({len(sentences) for sentences in found}) != 1:
        raise ValueError("FLORES languages must have the same number of sentences")
    return found


# https://universaldependencies.org/format.html
def surface_tokens(block: list[str]) -> list[tuple[str, int, bool]]:
    tokens = []
    covered = 0

    for line in block:
        fields = line.split("\t")

        index, form, upos, misc = (
            fields[0],
            fields[1],
            fields[3],
            fields[9],
        )
        spaced = "SpaceAfter=No" not in misc.split("|")
        if "-" in index:
            tokens.append((form, UNTAGGED, spaced))
            covered = int(index.split("-")[1])
        elif "." not in index and int(index) > covered:
            tokens.append((form, UPOS.index(upos), spaced))
    return tokens


def tagged_units(tokens: list[tuple[str, int, bool]], max_bytes: int) -> Tagged:
    groups: list[list[tuple[str, int]]] = [[]]
    for form, tag, spaced in tokens:
        groups[-1].append((form, tag))
        if spaced:
            groups.append([])
    text, tags = "", []
    for group in (group for group in groups if group):
        piece = (" " if text else "") + "".join(form for form, _ in group)
        units = to_units(piece, max_bytes)
        tag = group[0][1] if len(group) == 1 and len(units) == 1 else UNTAGGED
        text += piece
        tags += [tag] * len(units)
    return Tagged(text, tags)


def treebank(raw: bytes, max_bytes: int) -> list[Tagged]:
    sentences = []
    block: list[str] = []
    for line in raw.decode().splitlines():
        if not line.strip():
            if block:
                sentences.append(tagged_units(surface_tokens(block), max_bytes))
            block = []
        elif not line.startswith("#"):
            block.append(line)
    if block:
        sentences.append(tagged_units(surface_tokens(block), max_bytes))
    return sentences


# https://huggingface.co/datasets/jumelet/multiblimp
def minimal_pairs(raw: bytes) -> list[tuple[str, str]]:
    csv.field_size_limit(sys.maxsize)
    rows = csv.DictReader(
        io.StringIO(raw.decode()), delimiter="\t", quoting=csv.QUOTE_NONE
    )

    return [(row["sen"], row["wrong_sen"]) for row in rows]


@dataclass(frozen=True)
class Benchmarks:
    flores: list[list[str]]
    treebanks: dict[str, list[bytes]]
    pairs: list[list[tuple[str, str]]]


def fetch_benchmarks(spec: BenchmarkSources, languages: list[str]) -> Benchmarks:
    return Benchmarks(
        flores=flores_sentences(spec.flores, languages),
        treebanks={
            part: [checked(spec.treebanks[language][split]) for language in languages]
            for part, split in TREEBANK_PARTS.items()
        },
        pairs=[
            minimal_pairs(checked(spec.minimal_pairs[language]))
            for language in languages
        ],
    )
