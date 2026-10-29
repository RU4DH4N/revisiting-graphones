import math
from dataclasses import asdict, dataclass
from functools import partial

import numpy as np
import torch
from torch import Tensor

from data.benchmarks import BLIMP, FLORES, UD_DEV, UD_TEST, UNTAGGED, UPOS, half
from data.bilingual import BilingualCorpus, Encoded
from models.shared.numerics import DIVISION_FLOOR, NORM_FLOOR
from workers.evaluate import Scorer, Scores
from workers.records import EvaluationRecord


@dataclass(frozen=True)
class Measures:
    probe_steps: int
    probe_decay: float
    neighbours: int
    control_seed: int


@dataclass(frozen=True)
class PartOfSpeech:
    pos_accuracy: float
    pos_selectivity: float
    pos_transfer: float


@dataclass(frozen=True)
class Parallel:
    retrieval: float
    retrieval_csls: float
    cka: float


@dataclass(frozen=True)
class BenchmarkScores:
    minimal_pairs: float
    part_of_speech: PartOfSpeech
    parallel: Parallel


def in_half(part: str, documents: np.ndarray, split: str) -> np.ndarray:
    return np.array([half(f"{part}:{document}") == split for document in documents])


def minimal_pair_accuracy(scores: Scores, part: Encoded, split: str) -> float:
    totals = np.bincount(
        scores.documents, weights=scores.nats, minlength=len(part.languages)
    )
    good = np.arange(0, len(part.languages), 2)
    chosen = good[in_half(BLIMP, good, split)]
    correct = totals[chosen] < totals[chosen + 1]
    spoken = part.languages[chosen]
    return float(np.mean([correct[spoken == k].mean() for k in np.unique(spoken)]))


@dataclass(frozen=True)
class Probe:
    weights: Tensor
    mean: Tensor
    scale: Tensor

    def design(self, features: Tensor) -> Tensor:
        scaled = (features - self.mean) / self.scale
        return torch.cat((scaled, torch.ones_like(features[:, :1])), 1)

    def accuracy(self, features: Tensor, labels: Tensor) -> float:
        guesses = (self.design(features) @ self.weights).argmax(1)
        return float((guesses == labels).float().mean())


def probe_loss(
    optimiser: torch.optim.Optimizer,
    design: Tensor,
    weights: Tensor,
    labels: Tensor,
    decay: float,
) -> Tensor:
    optimiser.zero_grad()
    value = torch.nn.functional.cross_entropy(design @ weights, labels)
    value = value + decay * weights[:-1].square().sum()
    value.backward()
    return value


def fit_probe(features: Tensor, labels: Tensor, measures: Measures) -> Probe:
    probe = Probe(
        torch.zeros(features.shape[1] + 1, len(UPOS), requires_grad=True),
        features.mean(0),
        features.std(0).clamp_min(NORM_FLOOR),
    )
    design = probe.design(features)
    optimiser = torch.optim.LBFGS(
        [probe.weights], max_iter=measures.probe_steps, line_search_fn="strong_wolfe"
    )
    optimiser.step(
        partial(
            probe_loss, optimiser, design, probe.weights, labels, measures.probe_decay
        )
    )
    return Probe(probe.weights.detach(), probe.mean, probe.scale)


# https://arxiv.org/abs/1909.03368
def control_tags(types: np.ndarray, tags: np.ndarray, seed: int) -> dict[int, int]:
    rng = np.random.default_rng(seed)
    share = np.bincount(tags, minlength=len(UPOS)) / len(tags)
    unique = np.unique(types)
    drawn = rng.choice(len(UPOS), size=len(unique), p=share)
    return dict(zip(unique.tolist(), drawn.tolist(), strict=True))


@dataclass(frozen=True)
class Positions:
    features: np.ndarray
    tags: np.ndarray
    types: np.ndarray
    languages: np.ndarray


def tagged_positions(
    corpus: BilingualCorpus,
    part: str,
    messages: list[np.ndarray],
    documents: np.ndarray,
) -> Positions:
    encoded = corpus.parts[part]
    features, tags, types, languages = [], [], [], []
    for document, rows in zip(documents, messages, strict=True):
        start = encoded.offsets[document]
        found = corpus.tags[part][start : start + len(rows)]
        keep = found != UNTAGGED
        features.append(rows[keep])
        tags.append(found[keep].astype(np.int64))
        types.append(np.asarray(encoded.ids[start : start + len(rows)])[keep])
        languages.append(np.full(int(keep.sum()), encoded.languages[document]))
    return Positions(
        np.concatenate(features),
        np.concatenate(tags),
        np.concatenate(types),
        np.concatenate(languages),
    )


@dataclass(frozen=True)
class ProbeScores:
    within: float
    control: float
    across: list[float]


def probe_scores(
    train: Positions,
    test: Positions,
    source: int,
    languages: list[int],
    measures: Measures,
) -> ProbeScores:
    rows = train.languages == source
    own = test.languages == source
    features = torch.from_numpy(train.features[rows])
    probe = fit_probe(features, torch.from_numpy(train.tags[rows]), measures)
    shuffled = control_tags(
        np.concatenate((train.types[rows], test.types[own])),
        train.tags[rows],
        measures.control_seed,
    )
    noise = torch.tensor([shuffled[t] for t in train.types[rows].tolist()])
    control_probe = fit_probe(features, noise, measures)
    x, y = torch.from_numpy(test.features[own]), torch.from_numpy(test.tags[own])
    fake = torch.tensor([shuffled[t] for t in test.types[own].tolist()])
    across = [
        probe.accuracy(
            torch.from_numpy(test.features[test.languages == target]),
            torch.from_numpy(test.tags[test.languages == target]),
        )
        for target in languages
        if target != source
    ]
    return ProbeScores(probe.accuracy(x, y), control_probe.accuracy(x, fake), across)


def part_of_speech(
    corpus: BilingualCorpus,
    messages: dict[str, list[np.ndarray]],
    documents: dict[str, np.ndarray],
    languages: list[int],
    measures: Measures,
) -> PartOfSpeech:
    train = tagged_positions(corpus, UD_DEV, messages[UD_DEV], documents[UD_DEV])
    test = tagged_positions(corpus, UD_TEST, messages[UD_TEST], documents[UD_TEST])
    found = [
        probe_scores(train, test, source, languages, measures) for source in languages
    ]
    accuracy = float(np.mean([scores.within for scores in found]))
    across = float(np.mean([value for scores in found for value in scores.across]))
    return PartOfSpeech(
        pos_accuracy=accuracy,
        pos_selectivity=accuracy - float(np.mean([scores.control for scores in found])),
        pos_transfer=across / accuracy if accuracy > 0 else math.nan,
    )


def nearest_accuracy(similarity: np.ndarray) -> float:
    forward = similarity.argmax(1) == np.arange(len(similarity))
    backward = similarity.argmax(0) == np.arange(len(similarity))
    return float((forward.mean() + backward.mean()) / 2)


# https://arxiv.org/abs/1905.00414
def linear_cka(left: np.ndarray, right: np.ndarray) -> float:
    left = left - left.mean(0)
    right = right - right.mean(0)
    cross = np.linalg.norm(right.T @ left) ** 2
    return float(
        cross / (np.linalg.norm(left.T @ left) * np.linalg.norm(right.T @ right))
    )


# https://arxiv.org/abs/1710.04087
def parallel(
    sentences: list[np.ndarray], count: int, chosen: np.ndarray, neighbours: int
) -> Parallel:
    pooled = np.stack([rows.mean(0) for rows in sentences])
    left, right = pooled[chosen], pooled[count + chosen]
    unit = []
    for side in (left, right):
        side = side - side.mean(0)
        unit.append(
            side / np.linalg.norm(side, axis=1, keepdims=True).clip(DIVISION_FLOOR)
        )
    cosine = unit[0] @ unit[1].T
    k = min(neighbours, len(cosine) - 1)
    left_density = np.sort(cosine, 1)[:, -k:].mean(1)
    right_density = np.sort(cosine, 0)[-k:].mean(0)

    csls = 2 * cosine - left_density[:, None] - right_density[None, :]
    return Parallel(
        retrieval=nearest_accuracy(cosine),
        retrieval_csls=nearest_accuracy(csls),
        cka=linear_cka(left, right),
    )


def benchmark_scores(
    scorer: Scorer,
    corpus: BilingualCorpus,
    split: str,
    languages: list[int],
    measures: Measures,
) -> BenchmarkScores:
    pairs = scorer.score(corpus.parts[BLIMP], languages)
    documents = {
        part: np.flatnonzero(np.isin(corpus.parts[part].languages, languages))
        for part in (UD_DEV, UD_TEST, FLORES)
    }
    documents[UD_TEST] = documents[UD_TEST][in_half(UD_TEST, documents[UD_TEST], split)]
    messages = {
        part: scorer.represent(corpus.parts[part], chosen)
        for part, chosen in documents.items()
    }
    count = len(documents[FLORES]) // len(languages)
    sentences = np.arange(count)
    chosen = sentences[in_half(FLORES, sentences, split)]
    return BenchmarkScores(
        minimal_pairs=minimal_pair_accuracy(pairs, corpus.parts[BLIMP], split),
        part_of_speech=part_of_speech(corpus, messages, documents, languages, measures),
        parallel=parallel(messages[FLORES], count, chosen, measures.neighbours),
    )


def evaluation_record(
    bits_per_byte: float,
    by_language: dict[str, float],
    words: int,
    scores: BenchmarkScores,
) -> EvaluationRecord:
    return EvaluationRecord(
        bits_per_byte=bits_per_byte,
        minimal_pairs=scores.minimal_pairs,
        **asdict(scores.part_of_speech),
        **asdict(scores.parallel),
        by_language=by_language,
        words=words,
    )
