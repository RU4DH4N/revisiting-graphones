import itertools
import math
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np

from data.bilingual import BilingualCorpus
from data.splits import TRAIN
from framework.seeds import seed_for
from framework.storage import read_json, write_json
from models.shared.numerics import DIVISION_FLOOR
from models.shared.symbols import BLANK

ESPEAK = (1, 52, 0)
SOUND_MARK = "/"
PHONE_MARK = "|"

Alignment = dict[str, list[list[str]] | None]
Phonemiser = Callable[[str, list[str]], list[str]]


@dataclass(frozen=True)
class Costs:
    silent: float
    double: float
    extra: float
    pair: float
    unseen: float

    @cached_property
    def shapes(self) -> list[tuple[tuple[int, int], float]]:
        return [
            ((1, 0), self.silent),
            ((1, 1), 0.0),
            ((1, 2), self.double),
            ((0, 1), self.extra),
            ((2, 1), self.pair),
        ]


@dataclass(frozen=True)
class GraphoneSettings:
    sample: int
    rounds: int
    costs: Costs
    voices: dict[str, str]


def phones(ipa: str) -> list[str]:
    return [phone for word in ipa.split() for phone in word.split(PHONE_MARK) if phone]


def fold(core: str) -> str:
    return "".join(ch.lower() if len(ch.lower()) == 1 else ch for ch in core)


Steps = dict[tuple[int, int], tuple[int, int]]


def spoken_spans(sounds: list[str]) -> dict[tuple[int, int], str]:
    m = len(sounds)
    return {
        (j, b): " ".join(sounds[j:b])
        for j in range(m + 1)
        for b in range(j, min(m, j + 2) + 1)
    }


def best_paths(
    letters: str, sounds: list[str], cost: dict[tuple[str, str], float], costs: Costs
) -> tuple[list[list[float]], Steps]:
    n, m = len(letters), len(sounds)
    spoken = spoken_spans(sounds)
    best = [[math.inf] * (m + 1) for _ in range(n + 1)]
    back: Steps = {}
    best[0][0] = 0.0
    for i, j in itertools.product(range(n + 1), range(m + 1)):
        here = best[i][j]
        if here == math.inf:
            continue
        for (take, emit), penalty in costs.shapes:
            a, b = i + take, j + emit
            if a > n or b > m:
                continue
            key = (letters[i:a], spoken[(j, b)])
            value = here + cost.get(key, costs.unseen) + penalty
            if value < best[a][b]:
                best[a][b] = value
                back[(a, b)] = (i, j)
    return best, back


def traced(back: Steps, letters: str, sounds: list[str]) -> list[tuple[str, list[str]]]:
    path: list[tuple[str, list[str]]] = []
    a, b = len(letters), len(sounds)
    while (a, b) != (0, 0):
        i, j = back[(a, b)]
        path.append((letters[i:a], sounds[j:b]))
        a, b = i, j
    return path[::-1]


def grouped(path: list[tuple[str, list[str]]]) -> list[tuple[str, list[str]]]:
    found: list[tuple[str, list[str]]] = []
    lead: list[str] = []
    for group, said in path:
        if group:
            found.append((group, said))
        elif found:
            found[-1] = (found[-1][0], found[-1][1] + said)
        else:
            lead += said
    if found:
        found[0] = (found[0][0], lead + found[0][1])
    return found


# https://aclanthology.org/N07-1047/
def segments(
    letters: str, sounds: list[str], cost: dict[tuple[str, str], float], costs: Costs
) -> list[tuple[str, list[str]]] | None:
    best, back = best_paths(letters, sounds, cost, costs)
    reached = best[len(letters)][len(sounds)] != math.inf
    return grouped(traced(back, letters, sounds)) if reached else None


def align(
    letters: str, sounds: list[str], cost: dict[tuple[str, str], float], costs: Costs
) -> list[list[str]] | None:
    found = segments(letters, sounds, cost, costs)
    if found is None:
        return None
    return [said if k == 0 else [] for group, said in found for k in range(len(group))]


def refine(
    spoken: dict[str, list[str]], frequent: list[str], rounds: int, costs: Costs
) -> dict[tuple[str, str], float]:
    cost: dict[tuple[str, str], float] = {}
    for _ in range(rounds):
        tally: Counter[tuple[str, str]] = Counter()
        for core in frequent:
            for group, said in segments(core, spoken[core], cost, costs) or []:
                tally[(group, " ".join(said))] += 1
        totals: Counter[str] = Counter()
        for (group, _), n in tally.items():
            totals[group[0]] += n
        cost = {key: -math.log(n / totals[key[0][0]]) for key, n in tally.items()}
    return cost


def espeak(voice: str, words: list[str]) -> list[str]:
    from phonemizer.backend import EspeakBackend

    if EspeakBackend.version() != ESPEAK:
        raise RuntimeError(
            f"graphones need espeak-ng {ESPEAK}, found {EspeakBackend.version()}"
        )
    from phonemizer.separator import Separator

    backend = EspeakBackend(voice, with_stress=False, language_switch="remove-flags")
    separator = Separator(phone=PHONE_MARK, word=" ", syllable="")
    return backend.phonemize(words, separator=separator, strip=True)


def word_parts(corpus: BilingualCorpus) -> list[tuple[str, ...]]:
    texts = [
        bytes(corpus.table[t][: corpus.lengths[t]]).decode(errors="replace")
        for t in range(len(corpus.lengths))
    ]
    return [parts_of(text) for text in texts]


def word_char(ch: str) -> bool:
    return ch.isalnum() or ch == "_"


def parts_of(text: str) -> tuple[str, str, str, str]:
    space = len(text) - len(text.lstrip())
    start = space
    while start < len(text) and not word_char(text[start]):
        start += 1
    end = len(text)
    while end > start and not word_char(text[end - 1]):
        end -= 1
    return text[:space], text[space:start], text[start:end], text[end:]


def type_counts(corpus: BilingualCorpus) -> np.ndarray:
    part = corpus.parts[TRAIN]
    ids = np.asarray(part.ids)
    spoken = part.word_languages()
    return np.stack(
        [
            np.bincount(ids[spoken == k], minlength=len(corpus.lengths))
            for k in range(len(corpus.languages))
        ]
    )


def type_presence(corpus: BilingualCorpus) -> np.ndarray:
    seen = np.zeros((len(corpus.languages), len(corpus.lengths)), dtype=bool)
    for part in corpus.parts.values():
        ids = np.asarray(part.ids)
        spoken = part.word_languages()
        for k in range(len(corpus.languages)):
            seen[k, ids[spoken == k]] = True
    return seen


def core_counts(
    parts: list[tuple[str, ...]], counts: np.ndarray, spoken: dict[str, list[str]]
) -> Counter[str]:
    weights: Counter[str] = Counter()
    for t in np.flatnonzero(counts):
        core = fold(parts[t][2])
        if core in spoken:
            weights[core] += int(counts[t])
    return weights


def align_language(
    parts: list[tuple[str, ...]],
    counts: np.ndarray,
    present: np.ndarray,
    voice: str,
    sample: int,
    rounds: int,
    costs: Costs,
    phonemise: Phonemiser,
) -> Alignment:
    cores = sorted(
        {
            fold(parts[t][2])
            for t in np.flatnonzero(present)
            if parts[t][2] and not any(ch.isdigit() for ch in parts[t][2])
        }
    )
    spoken = {
        core: phones(s) for core, s in zip(cores, phonemise(voice, cores), strict=True)
    }
    weights = core_counts(parts, counts, spoken)
    frequent = [core for core, _ in weights.most_common(sample)]
    cost = refine(spoken, frequent, rounds, costs)
    return {core: align(core, spoken[core], cost, costs) for core in spoken}


def align_corpus(
    corpus: BilingualCorpus,
    parts: list[tuple[str, ...]],
    counts: np.ndarray,
    settings: GraphoneSettings,
    phonemise: Phonemiser,
) -> list[Alignment]:
    present = type_presence(corpus)
    return [
        align_language(
            parts,
            counts[k],
            present[k],
            settings.voices[language],
            settings.sample,
            settings.rounds,
            settings.costs,
            phonemise,
        )
        for k, language in enumerate(corpus.languages)
    ]


@dataclass(frozen=True)
class Units:
    codes: np.ndarray
    lengths: np.ndarray
    keys: list[str]


class UnitIndex:
    def __init__(self) -> None:
        self.rows: dict[str, int] = {}

    def row(self, letter: str, sound: str | None) -> int:
        key = letter if sound is None else f"{letter}{SOUND_MARK}{sound}"
        if key not in self.rows:
            found = len(self.rows)
            self.rows[key] = found if found < BLANK else found + 1
        return self.rows[key]

    def word(self, parts: tuple[str, ...], sounds: list[list[str]] | None) -> list[int]:
        space, before, core, after = parts
        units = [self.row(ch, None) for ch in space + before]
        if sounds is None:
            units += [self.row(ch, None) for ch in core]
        else:
            units += [
                self.row(ch, "".join(s)) for ch, s in zip(core, sounds, strict=True)
            ]
        return units + [self.row(ch, None) for ch in after]


def build_units(
    parts: list[tuple[str, ...]], alignments: list[Alignment], width: int
) -> Units:
    index = UnitIndex()
    codes = np.full((len(alignments), len(parts), width), BLANK, dtype=np.int64)
    lengths = np.zeros((len(alignments), len(parts)), dtype=np.int64)
    for k, aligned in enumerate(alignments):
        for t, word in enumerate(parts):
            core = word[2]
            units = index.word(word, aligned.get(fold(core)) if core else None)[:width]
            codes[k, t, : len(units)] = units
            lengths[k, t] = len(units)
    rows = max(max(index.rows.values()) + 1, BLANK + 1)
    keys = [""] * rows
    for key, row in index.rows.items():
        keys[row] = key
    shifted = codes + (np.arange(len(alignments)) * rows)[:, None, None]
    return Units(np.where(codes == BLANK, BLANK, shifted), lengths, keys)


def split_key(key: str) -> tuple[str, str | None]:
    sounded = len(key) > 1 and key[1] == SOUND_MARK
    return (key[0], key[2:]) if sounded else (key, None)


def has_sound(keys: list[str]) -> np.ndarray:
    return np.array([bool(key) and split_key(key)[1] is not None for key in keys])


def stream_rows(
    keys: list[str], languages: int, dims: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    letters: dict[str, int] = {}
    sounds: dict[str, int] = {}
    letter_of = np.full(len(keys), -1, dtype=np.int64)
    sound_of = np.full(len(keys), -1, dtype=np.int64)
    for row, key in enumerate(keys):
        if not key:
            continue
        letter, sound = split_key(key)
        letter_of[row] = letters.setdefault(letter, len(letters))
        if sound is not None:
            sound_of[row] = sounds.setdefault(sound, len(sounds))
    generator = np.random.default_rng(seed)
    signs = np.array([-1.0, 1.0], dtype=np.float32)
    letter_rows = generator.choice(signs, size=(len(letters), dims))
    sound_rows = generator.choice(signs, size=(len(sounds), dims))
    has = has_sound(keys)
    lettered = np.zeros((len(keys), dims), dtype=np.int8)
    sounded = np.zeros((len(keys), dims), dtype=np.int8)
    lettered[letter_of >= 0] = letter_rows[letter_of[letter_of >= 0]]
    sounded[has] = sound_rows[sound_of[has]]
    return np.concatenate([lettered] * languages), np.concatenate([sounded] * languages)


def joint_rows(keys: list[str], languages: int, dims: int, seed: int) -> np.ndarray:
    _, joint_of = np.unique(keys, return_inverse=True)
    rows = np.random.default_rng(seed).choice(
        np.array([-1, 1], dtype=np.int8), size=(joint_of.max() + 1, dims)
    )[joint_of]
    return np.concatenate([rows] * languages)


def entropy(table: np.ndarray) -> np.ndarray:
    share = table / table.sum(1, keepdims=True).clip(DIVISION_FLOOR)
    return -(share * np.log2(share.clip(DIVISION_FLOOR))).sum(1)


def features(units: Units, counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    per = len(units.keys)
    split = [split_key(key) for key in units.keys]
    letter_names = np.unique([letter for letter, _ in split], return_inverse=True)[1]
    sound_names = np.unique([sound or "" for _, sound in split], return_inverse=True)[1]
    sounded = has_sound(units.keys)
    silent = np.array([sound == "" for _, sound in split], dtype=np.float32)
    languages = units.codes.shape[0]
    table = np.zeros((languages * per, 3), dtype=np.float32)
    uses = np.zeros(languages * per, dtype=np.float64)
    for k in range(languages):
        weights = counts[k][: units.lengths.shape[1]].astype(np.float64)
        local = units.codes[k] - k * per
        inside = np.arange(local.shape[1])[None, :] < units.lengths[k][:, None]
        used = np.bincount(
            local[inside],
            weights=np.broadcast_to(weights[:, None], local.shape)[inside],
            minlength=per,
        )[:per]
        uses[k * per : (k + 1) * per] = used
        for column, (given, other) in enumerate(
            ((letter_names, sound_names), (sound_names, letter_names))
        ):
            joint = np.zeros((given.max() + 1, other.max() + 1))
            np.add.at(joint, (given[sounded], other[sounded]), used[sounded])
            table[k * per : (k + 1) * per, column] = np.where(
                sounded, entropy(joint)[given], 0.0
            )
        table[k * per : (k + 1) * per, 2] = silent
    return table, uses


@dataclass(frozen=True)
class Tables:
    keys: list[str]
    codes: np.ndarray
    lengths: np.ndarray
    has: np.ndarray
    features: np.ndarray
    uses: np.ndarray
    root: Path

    def codes_for(self, dims: int) -> tuple[np.ndarray, ...]:
        return tuple(np.load(self.root / name) for name in code_files(dims))

    def table(self, name: str) -> np.ndarray:
        return np.load(self.root / table_file(name))


FILES = ("keys.json", "codes.npy", "lengths.npy", "has.npy", "features.npy", "uses.npy")
ALIGNMENT_FILE = "alignment.json"


def code_files(dims: int) -> tuple[str, str, str]:
    return (f"letters-{dims}.npy", f"sounds-{dims}.npy", f"joint-{dims}.npy")


def table_file(name: str) -> str:
    return f"mix-{name}.npy"


def write_tables(
    root: Path, units: Units, counts: np.ndarray, widths: list[int], root_seed: int
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    languages = units.codes.shape[0]
    table, uses = features(units, counts)
    write_json(root / "keys.json", units.keys)
    np.save(
        root / "codes.npy",
        units.codes.astype(np.min_scalar_type(units.codes.max())),
    )
    np.save(root / "lengths.npy", units.lengths.astype(np.int16))
    np.save(root / "features.npy", table)
    np.save(root / "uses.npy", uses)
    for dims in widths:
        letters, sounds = stream_rows(
            units.keys, languages, dims, seed_for(root_seed, "stream-codes")
        )
        joint = joint_rows(
            units.keys, languages, dims, seed_for(root_seed, "joint-codes")
        )
        for name, rows in zip(code_files(dims), (letters, sounds, joint), strict=True):
            np.save(root / name, rows)
    np.save(root / "has.npy", np.concatenate([has_sound(units.keys)] * languages))


def read_tables(root: Path) -> Tables:
    return Tables(
        read_json(root / "keys.json"),
        np.load(root / "codes.npy").astype(np.int64),
        np.load(root / "lengths.npy").astype(np.int64),
        np.load(root / "has.npy"),
        np.load(root / "features.npy"),
        np.load(root / "uses.npy"),
        root,
    )
