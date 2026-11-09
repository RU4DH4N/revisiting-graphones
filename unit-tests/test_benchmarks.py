import hashlib
import math
from dataclasses import asdict

import numpy as np
import pytest
import torch
from conftest import Tiny, config
from hypothesis import given, strategies as st
from strategies import hypothesis_settings

import data.benchmarks
from data.benchmarks import (
    HALVES,
    UNTAGGED,
    UPOS,
    Download,
    checked,
    half,
    minimal_pairs,
    treebank,
)
from data.bilingual import Encoded
from data.corpus import to_units
from workers.benchmarks import (
    Measures,
    control_tags,
    fit_probe,
    in_half,
    linear_cka,
    minimal_pair_accuracy,
    nearest_accuracy,
    parallel,
    part_of_speech,
    translations,
)
from workers.evaluate import Scores
from workers.settings import parse

EVALUATION = parse(config()).evaluation
MEASURES = Measures(
    EVALUATION.probe_steps, EVALUATION.probe_decay, EVALUATION.neighbours, 7
)

SENTENCE = """# text = Je sens qu'il pleut du vin.
1\tJe\tje\tPRON\t_\t_\t2\tnsubj\t_\t_
2\tsens\tsentir\tVERB\t_\t_\t0\troot\t_\t_
3\tqu'\tque\tSCONJ\t_\t_\t5\tmark\t_\tSpaceAfter=No
4\til\til\tPRON\t_\t_\t5\tnsubj\t_\t_
5\tpleut\tpleuvoir\tVERB\t_\t_\t2\tccomp\t_\t_
6-7\tdu\t_\t_\t_\t_\t_\t_\t_\t_
6\tde\tde\tADP\t_\t_\t8\tcase\t_\t_
7\tle\tle\tDET\t_\t_\t8\tdet\t_\t_
8\tvin\tvin\tNOUN\t_\t_\t5\tobl\t_\tSpaceAfter=No
9\t.\t.\tPUNCT\t_\t_\t2\tpunct\t_\t_

"""


def test_a_treebank_sentence_becomes_its_surface_text_with_one_tag_per_unit() -> None:
    (found,) = treebank(SENTENCE.encode(), 16)
    assert found.text == "Je sens qu'il pleut du vin."
    names = [UPOS[tag] if tag != UNTAGGED else None for tag in found.tags]
    assert names == ["PRON", "VERB", None, "VERB", None, None]


def misc(spaced: bool) -> str:
    return "_" if spaced else "SpaceAfter=No"


@given(
    st.lists(
        st.tuples(
            st.text(alphabet="abcdéж'.", min_size=1, max_size=12),
            st.sampled_from(UPOS),
            st.booleans(),
        ),
        min_size=1,
        max_size=12,
    ),
    st.integers(min_value=4, max_value=10),
)
@hypothesis_settings
def test_tags_always_line_up_with_the_units_of_the_text(
    tokens: list[tuple[str, str, bool]], max_bytes: int
) -> None:
    lines = [
        f"{i}\t{form}\t_\t{upos}\t_\t_\t0\tdep\t_\t{misc(spaced)}"
        for i, (form, upos, spaced) in enumerate(tokens, start=1)
    ]
    (found,) = treebank(("\n".join(lines) + "\n\n").encode(), max_bytes)
    assert len(found.tags) == len(to_units(found.text, max_bytes))


def test_minimal_pairs_read_the_grammatical_and_wrong_sentences() -> None:
    raw = 'sen\twrong_sen\tother\nThe dog runs.\tThe dog run.\t{"a": 1}\n'.encode()
    assert minimal_pairs(raw) == [("The dog runs.", "The dog run.")]


def test_halves_are_fixed_and_split_items_roughly_evenly() -> None:
    keys = [f"item:{i}" for i in range(2000)]
    assert [half(key) for key in keys] == [half(key) for key in keys]
    share = np.mean([half(key) == HALVES[0] for key in keys])
    assert 0.45 < share < 0.55


@given(st.integers(min_value=0, max_value=1000))
@hypothesis_settings
def test_cka_is_one_for_the_same_space_up_to_rotation_and_scale(seed: int) -> None:
    rng = np.random.default_rng(seed)
    left = rng.normal(size=(40, 6))
    rotation, _ = np.linalg.qr(rng.normal(size=(6, 6)))
    assert linear_cka(left, 3.0 * left @ rotation + 2.0) == pytest.approx(1.0)


def test_cka_is_low_for_unrelated_spaces() -> None:
    rng = np.random.default_rng(0)
    assert linear_cka(rng.normal(size=(500, 8)), rng.normal(size=(500, 8))) < 0.1


def test_identical_translations_are_retrieved_and_shuffled_ones_are_not() -> None:
    rng = np.random.default_rng(1)
    sentences = [rng.normal(size=(3, 16)) for _ in range(50)]
    found = parallel(sentences + sentences, 50, np.arange(50), MEASURES.neighbours)
    assert asdict(found) == pytest.approx(
        {"retrieval": 1.0, "retrieval_csls": 1.0, "cka": 1.0}
    )
    shuffled = [sentences[i] for i in rng.permutation(50)]
    shuffled_found = parallel(
        sentences + shuffled, 50, np.arange(50), MEASURES.neighbours
    )
    assert shuffled_found.retrieval < 0.2


def test_a_probe_reads_separable_tags_and_its_control_cannot() -> None:
    rng = np.random.default_rng(2)
    tags = rng.integers(0, 4, 400)
    features = torch.from_numpy(
        (np.eye(8)[tags] * 4 + rng.normal(size=(400, 8))).astype(np.float32)
    )
    labels = torch.from_numpy(tags)
    assert fit_probe(features, labels, MEASURES).accuracy(features, labels) > 0.95
    types = rng.integers(0, 200, 400)
    shuffled = control_tags(types, tags, 7)
    assert shuffled == control_tags(types, tags, 7)
    noise = torch.tensor([shuffled[t] for t in types.tolist()])
    held = slice(300, 400)
    control = fit_probe(features[:300], noise[:300], MEASURES)
    assert control.accuracy(features[held], noise[held]) < 0.6


def test_a_minimal_pair_counts_when_the_grammatical_sentence_costs_less() -> None:
    part = Encoded(
        np.zeros(8, dtype=np.int32),
        np.arange(0, 9, 2),
        np.array([0, 0, 1, 1], dtype=np.int8),
    )
    documents = np.array([0, 1, 2, 3])
    for nats, expected in (
        ([1.0, 2.0, 1.0, 2.0], 1.0),
        ([2.0, 1.0, 2.0, 1.0], 0.0),
        ([1.0, 1.0, 1.0, 1.0], 0.0),
    ):
        scores = Scores(
            types=documents,
            languages=part.languages[documents],
            nats=np.array(nats),
            symbols=np.ones(4),
            documents=documents,
        )
        found = [minimal_pair_accuracy(scores, part, split) for split in HALVES]
        assert any(not math.isnan(value) for value in found)
        assert all(math.isnan(value) or value == expected for value in found)


def test_bits_per_byte_counts_spelled_bytes_but_not_the_stop() -> None:
    scores = Scores(
        types=np.zeros(2),
        languages=np.zeros(2),
        nats=np.array([math.log(2) * 3, math.log(2) * 5]),
        symbols=np.array([3, 5]),
        documents=np.zeros(2),
    )
    assert scores.bits_per_byte(np.ones(2, dtype=bool)) == pytest.approx(8 / 6)


def test_csls_matches_a_direct_computation() -> None:
    rng = np.random.default_rng(3)
    count, k = 30, 3
    left = [rng.normal(size=(2, 6)) for _ in range(count)]
    right = [rng.normal(size=(2, 6)) for _ in range(count)]
    found = parallel(left + right, count, np.arange(count), k)
    sides = []
    for side in (left, right):
        pooled = np.stack([rows.mean(0) for rows in side])
        pooled -= pooled.mean(0)
        sides.append(pooled / np.linalg.norm(pooled, axis=1, keepdims=True))
    cosine = sides[0] @ sides[1].T
    near_left = [np.mean(sorted(cosine[i])[-k:]) for i in range(count)]
    near_right = [np.mean(sorted(cosine[:, j])[-k:]) for j in range(count)]
    csls = np.array(
        [
            [2 * cosine[i, j] - near_left[i] - near_right[j] for j in range(count)]
            for i in range(count)
        ]
    )
    assert found.retrieval_csls == pytest.approx(nearest_accuracy(csls))


def test_each_half_scores_only_its_own_minimal_pairs() -> None:
    pairs = 40
    part = Encoded(
        np.zeros(4 * pairs, dtype=np.int32),
        np.arange(0, 4 * pairs + 1, 2),
        np.zeros(2 * pairs, dtype=np.int8),
    )
    good = np.arange(0, 2 * pairs, 2)
    right = in_half("blimp", good, "selection")
    nats = np.zeros(2 * pairs)
    nats[good + 1] = np.where(right, 2.0, 0.0)
    documents = np.arange(2 * pairs)
    scores = Scores(documents, part.languages, nats, np.ones(2 * pairs), documents)
    assert 0 < right.sum() < pairs
    assert minimal_pair_accuracy(scores, part, "selection") == 1.0
    assert minimal_pair_accuracy(scores, part, "reporting") == 0.0


def test_retrieval_averages_both_directions() -> None:
    similarity = np.array([[1.0, 0.9], [0.0, 0.5]])
    assert nearest_accuracy(similarity) == pytest.approx(0.75)


def test_control_tags_follow_the_tag_frequencies() -> None:
    tags = np.array([0] * 900 + [3] * 100)
    drawn = control_tags(np.arange(2000), tags, 1)
    assert set(drawn.values()) <= {0, 3}
    assert np.mean([tag == 0 for tag in drawn.values()]) == pytest.approx(0.9, abs=0.03)


def tagged_messages(
    tiny: Tiny, documents: dict[str, np.ndarray], rng: np.random.Generator
) -> dict[str, list[np.ndarray]]:
    messages = {}
    for part, chosen in documents.items():
        encoded, tags = tiny.corpus.parts[part], tiny.corpus.tags[part]
        rows = []
        for document in chosen:
            span = slice(encoded.offsets[document], encoded.offsets[document + 1])
            found = tags[span].clip(0)
            noisy = np.eye(len(UPOS))[found] * 2.0 + rng.normal(
                size=(len(found), len(UPOS))
            )
            rows.append(noisy.astype(np.float32))
        messages[part] = rows
    return messages


def test_tags_written_into_the_messages_are_read_back_in_both_languages(
    tiny: Tiny,
) -> None:
    rng = np.random.default_rng(0)
    documents = {
        part: np.flatnonzero(np.isin(tiny.corpus.parts[part].languages, [0, 1]))
        for part in ("ud-dev", "ud-test")
    }
    messages = tagged_messages(tiny, documents, rng)
    found = part_of_speech(tiny.corpus, messages, documents, [0, 1], MEASURES)
    assert 0.6 < found.pos_accuracy < 0.85
    assert 0 < found.pos_selectivity < found.pos_accuracy
    assert found.pos_transfer > 0.85


def test_messages_that_only_name_the_word_have_no_selectivity_or_transfer(
    tiny: Tiny,
) -> None:
    rng = np.random.default_rng(0)
    documents = {
        part: np.flatnonzero(np.isin(tiny.corpus.parts[part].languages, [0, 1]))
        for part in ("ud-dev", "ud-test")
    }
    vocab = max(int(np.max(tiny.corpus.parts[part].ids)) for part in documents) + 1
    codes = rng.normal(size=(vocab, 32))
    messages = {}
    for part, chosen in documents.items():
        encoded = tiny.corpus.parts[part]
        rows = []
        for document in chosen:
            span = slice(encoded.offsets[document], encoded.offsets[document + 1])
            ids = np.asarray(encoded.ids[span])
            noisy = codes[ids] * 3 + rng.normal(size=(len(ids), 32))
            rows.append(noisy.astype(np.float32))
        messages[part] = rows
    found = part_of_speech(tiny.corpus, messages, documents, [0, 1], MEASURES)
    assert found.pos_accuracy > 0.9
    assert abs(found.pos_selectivity) < 0.1
    assert found.pos_transfer < 0.5


def test_one_language_has_part_of_speech_but_no_transfer_or_translations(
    tiny: Tiny,
) -> None:
    rng = np.random.default_rng(0)
    documents = {
        part: np.flatnonzero(tiny.corpus.parts[part].languages == 0)
        for part in ("ud-dev", "ud-test")
    }
    messages = tagged_messages(tiny, documents, rng)
    found = part_of_speech(tiny.corpus, messages, documents, [0], MEASURES)
    assert 0.5 < found.pos_accuracy and found.pos_selectivity > 0
    assert math.isnan(found.pos_transfer)
    missing = translations(messages, documents, [0], "selection", MEASURES)
    assert all(math.isnan(value) for value in asdict(missing).values())


class Reply:
    content = b"downloaded"

    def raise_for_status(self) -> None:
        return None


def replied(method: str, url: str, follow_redirects: bool) -> Reply:
    return Reply()


def test_a_download_that_does_not_match_its_pinned_hash_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(data.benchmarks, "http_backoff", replied)
    pinned = hashlib.sha256(b"downloaded").hexdigest()
    assert checked(Download("https://example.org/a", pinned)) == b"downloaded"
    with pytest.raises(ValueError, match="does not match its pinned hash"):
        checked(Download("https://example.org/a", "0" * 64))
