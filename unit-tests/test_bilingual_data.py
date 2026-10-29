import itertools
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from conftest import EN, FR, Tiny, benchmarks, config, synthetic
from hypothesis import given, strategies as st
from strategies import hypothesis_settings

import data.bilingual as bilingual
from data.benchmarks import BLIMP, Benchmarks
from data.bilingual import (
    BilingualCorpus,
    Vocabulary,
    collect,
    split_of,
    unit_bytes,
)
from data.splits import SPLITS, TRAIN
from models.graphone import TABLE
from workers import design, estimates
from workers.settings import parse, read_config

ROOT = Path(__file__).resolve().parents[1]


def test_split_assignment_is_a_pure_function_of_the_text() -> None:
    shares = {"train": 0.8, "selection": 0.1, "reporting": 0.1}
    assert split_of("some text", shares) == split_of("some text", shares)
    found = {split_of(f"document {i}", shares) for i in range(200)}
    assert found == set(SPLITS)


@given(st.lists(st.integers(min_value=1, max_value=4000), min_size=1, max_size=40))
@hypothesis_settings
def test_collection_fills_every_byte_target_to_within_a_word_and_never_past_it(
    sizes: list[int],
) -> None:
    texts = (f"doc {i} " + "word " * sizes[i % len(sizes)] for i in range(100_000))
    targets = {"train": 3000, "selection": 500, "reporting": 500}
    found = collect(texts, targets)
    for split in SPLITS:
        used = sum(len(t.encode()) for t in found[split])
        assert targets[split] - len("doc 99999 ") <= used <= targets[split]


def test_the_cut_document_ends_on_a_word_boundary() -> None:
    found = collect(
        (f"text {i} " + "café au lait " * 200 for i in range(1000)),
        {"train": 300, "selection": 100, "reporting": 100},
    )
    for texts in found.values():
        assert texts[-1].endswith(("lait", "au", "café"))


def test_collection_reports_an_exhausted_source() -> None:
    with pytest.raises(ValueError, match="ran out"):
        collect(iter(["short"]), {"train": 100, "selection": 100, "reporting": 100})


def test_vocabulary_round_trips_every_unit() -> None:
    vocabulary = Vocabulary(8)
    ids = vocabulary.encode(" café au lait, très bien")
    table, lengths = vocabulary.table()
    text = "".join(bytes(table[i, : lengths[i]]).decode() for i in ids)
    assert text == " café au lait, très bien"


def test_built_corpus_is_balanced_and_verified(tiny: Tiny) -> None:
    corpus = tiny.corpus
    words = corpus.metadata["statistics"][TRAIN]["words_by_language"]
    assert words["en"] > 0 and words["fr"] > 0
    assert int(np.asarray(corpus.parts[TRAIN].ids).max()) < len(corpus.lengths)


def test_the_config_names_every_field_exactly(tiny: Tiny) -> None:
    parse(tiny.raw)
    extra = {**tiny.raw, "data": {**tiny.raw["data"], "split": "words"}}
    with pytest.raises(ValueError, match="unknown \\['split'\\]"):
        parse(extra)


def test_only_three_streams_take_a_table_and_an_equal_mix_cannot_be_held(
    tiny: Tiny,
) -> None:
    arms = tiny.raw["arms"]
    for change, message in (
        ({**arms["baseline"], "mix": TABLE, "table": "avg"}, "needs three streams"),
        ({**arms["triple-direct"], "hold": True}, "cannot be held"),
        ({**arms["learned-direct"], "mix": "fixed"}, "mix must be one of"),
    ):
        broken = {**tiny.raw, "arms": {**arms, "broken": change}}
        with pytest.raises(ValueError, match=message):
            parse(broken)


def test_only_an_arm_whose_code_feeds_the_trunk_directly_is_matched(tiny: Tiny) -> None:
    arms = tiny.raw["arms"]
    broken = {
        **tiny.raw,
        "arms": {**arms, "broken": {**arms["baseline"], "matched": True}},
    }
    with pytest.raises(ValueError, match="only an arm without a projection"):
        parse(broken)


@pytest.mark.parametrize("pair", ["en-fr", "en-ru", "es-pt", "smoke"])
def test_every_arm_has_the_same_trunk_and_only_its_input_layer_adds_to_it(
    pair: str,
) -> None:
    settings = read_config(ROOT / "config" / f"{pair}.json")
    shape = estimates.plan(settings).trunk
    body = design.shaped_trunk_count(shape, shape.d_model)
    hidden = round(shape.mlp_ratio * shape.d_model)
    for arm in settings.arms.values():
        linear = arm.code_width * shape.d_model if arm.projection else 0
        mlp = arm.code_width * hidden + hidden * shape.d_model
        extra = mlp if arm.input_mlp else linear
        assert estimates.trunk_parameters(settings, arm) == body + extra


@pytest.mark.parametrize("width", [16, 48, 64, 96, 130, 420, 448, 512])
def test_heads_split_the_width_evenly_with_head_widths_nearest_the_convention(
    width: int,
) -> None:
    model = parse(config()).model
    heads = model.heads_for(width)
    assert width % heads == 0 and width // heads % 2 == 0
    if width % model.head_dimensions == 0:
        assert width // heads == model.head_dimensions


def documents(corpus: BilingualCorpus, split: str) -> set[bytes]:
    part = corpus.parts[split]
    ids = np.asarray(part.ids)
    return {
        b"".join(
            bytes(corpus.table[t][: corpus.lengths[t]])
            for t in ids[part.offsets[d] : part.offsets[d + 1]]
        )
        for d in range(len(part.languages))
    }


def test_no_document_appears_in_two_splits(tiny: Tiny) -> None:
    found = {split: documents(tiny.corpus, split) for split in SPLITS}
    assert all(found.values())
    for split, other in itertools.combinations(SPLITS, 2):
        assert not found[split] & found[other]


def test_the_unit_width_covers_the_chosen_share_of_words() -> None:
    documents = [" ".join(["ab"] * 9 + ["abcdefghijkl"])]
    assert unit_bytes(documents, 0.9, 1) == 3
    assert unit_bytes(documents, 1.0, 1) == 13
    assert unit_bytes(documents, 0.9, 5) == 5


def test_steps_cover_the_passes_over_the_training_words() -> None:
    settings = parse(
        {**config(), "passes": 2.0, "batch_windows": 4, "context_words": 6}
    )
    assert design.steps(settings, 1200) == 100
    assert design.steps(settings, 1201) == 101


def built_with(root: Path, extra: Benchmarks) -> BilingualCorpus:
    sources = iter([("en", synthetic(EN, 1, 400)), ("fr", synthetic(FR, 2, 400))])
    bilingual.build(parse(config()).data, root, sources, extra)
    return BilingualCorpus(root)


def texts_of(corpus: BilingualCorpus, part: str) -> list[str]:
    encoded = corpus.parts[part]
    return [
        "".join(
            bytes(corpus.table[unit, : corpus.lengths[unit]]).decode()
            for unit in encoded.ids[start:stop]
        )
        for start, stop in zip(encoded.offsets[:-1], encoded.offsets[1:], strict=True)
    ]


def test_a_pair_with_a_one_word_sentence_is_dropped_whole(tmp_path: Path) -> None:
    found = benchmarks()
    pairs = [[("cat", "the cat sat"), *found.pairs[0]], found.pairs[1]]
    corpus = built_with(tmp_path, replace(found, pairs=pairs))
    expected = [text for language in found.pairs for pair in language for text in pair]
    assert texts_of(corpus, BLIMP) == expected


def test_a_one_word_treebank_sentence_is_dropped_with_its_tags(tmp_path: Path) -> None:
    found = benchmarks()
    single = b"1\tcat\tcat\tNOUN\t_\t_\t0\troot\t_\t_\n\n"
    treebanks = {
        part: [single + raws[0], raws[1]] for part, raws in found.treebanks.items()
    }
    corpus = built_with(tmp_path, replace(found, treebanks=treebanks))
    for part in treebanks:
        assert "cat" not in texts_of(corpus, part)
        assert len(corpus.tags[part]) == len(corpus.parts[part].ids)


def test_a_flores_sentence_too_short_to_score_is_refused(tmp_path: Path) -> None:
    found = benchmarks()
    flores = [["cat", *found.flores[0][1:]], found.flores[1]]
    with pytest.raises(ValueError, match="FLORES sentence"):
        built_with(tmp_path, replace(found, flores=flores))


def test_a_corpus_file_that_changed_is_refused(tmp_path: Path) -> None:
    built_with(tmp_path, benchmarks())
    path = tmp_path / "train-languages.npy"
    values = np.load(path)
    np.save(path, 1 - values)
    with pytest.raises(ValueError, match="does not match its hash: train-languages"):
        BilingualCorpus(tmp_path)


def test_the_receiver_body_is_the_configured_share_of_the_trunk() -> None:
    settings = read_config(ROOT / "config" / "en-fr.json")
    model = settings.model
    width, layers = design.receiver_body(settings.trunk_parameters, 24, model)
    built = design.build_receiver(design.shape(width, layers, 25, width, model), model)
    channels = sum(p.numel() for p in built.channels.parameters())
    body = sum(p.numel() for p in built.parameters()) - channels
    assert design.body_count(width, layers, 24, model) == body
    target = settings.trunk_parameters * model.receiver_fraction
    assert abs(math.log(body / target)) < 0.1
