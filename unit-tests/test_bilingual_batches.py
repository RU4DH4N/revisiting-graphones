from dataclasses import replace

import numpy as np
import pytest
import torch
from conftest import Tiny, model_with, with_arm

from data.bilingual import Encoded
from models.bilingual import IGNORE, PAD, START, STOP
from models.graphone import BYTES, SPLIT
from workers.arms import expander_for
from workers.batches import Expander, WindowSampler, evaluation_rows


def expander(tiny: Tiny, device: torch.device) -> Expander:
    return Expander(tiny.corpus, tiny.tables.codes, tiny.tables.lengths, device)


def test_expanded_targets_spell_each_word_then_stop(tiny: Tiny) -> None:
    corpus = tiny.corpus
    rows = WindowSampler(corpus.parts["train"], [0, 1], 6, 4, 0).draw()
    batch = expander(tiny, torch.device("cpu"))(rows)
    targets = rows.words[:, 1:].reshape(-1)
    for index in range(len(targets)):
        word = int(targets[index])
        length = int(corpus.lengths[word])
        expected = list(corpus.table[word, :length]) + [STOP]
        assert batch.targets[index, : length + 1].tolist() == expected
        assert (batch.targets[index, length + 1 :] == IGNORE).all()
        assert batch.inputs[index, 0] == START
        assert batch.inputs[index, 1 : length + 1].tolist() == expected[:-1]


@pytest.mark.parametrize("streams", [SPLIT, BYTES])
def test_the_context_holds_each_words_units_in_its_own_language(
    tiny: Tiny, streams: str
) -> None:
    arm = replace(tiny.settings.arms["baseline"], streams=streams)
    rows = WindowSampler(tiny.corpus.parts["train"], [0, 1], 6, 8, 1).draw()
    context = expander_for(tiny.corpus, tiny.tables, arm, torch.device("cpu"))(
        rows
    ).context
    for window in range(len(rows.words)):
        language = int(rows.languages[window])
        for event in range(context.shape[1]):
            word = int(rows.words[window, event])
            if streams == BYTES:
                size = int(tiny.corpus.lengths[word])
                expected = list(tiny.corpus.table[word, :size])
            else:
                size = int(tiny.tables.lengths[language, word])
                expected = list(tiny.tables.codes[language, word, :size])
            assert context[window, event, :size].tolist() == expected
            assert (context[window, event, size:] == PAD).all()


def test_a_bytes_model_scores_a_batch_of_byte_units(tiny: Tiny) -> None:
    settings = with_arm(tiny.settings, "baseline", streams=BYTES)
    model = model_with(tiny, settings, "baseline")
    rows = WindowSampler(tiny.corpus.parts["train"], [0, 1], 6, 4, 2).draw()
    found = expander_for(
        tiny.corpus, tiny.tables, settings.arms["baseline"], torch.device("cpu")
    )
    batch = found(rows)
    with torch.no_grad():
        nll = model.word_nll(batch.inputs, batch.targets, model.messages(batch.context))
    assert torch.isfinite(nll).all() and nll.sum() > 0


def test_evaluation_scores_every_word_but_the_first_exactly_once(tiny: Tiny) -> None:
    part = tiny.corpus.parts["selection"]
    scored = np.zeros(len(part.languages), dtype=np.int64)
    for rows in evaluation_rows(part, [0, 1], 6, 4, 0):
        np.add.at(scored, rows.documents.numpy(), rows.scored.sum(1).numpy())
    assert (scored == np.diff(part.offsets) - 1).all()


def test_sampled_windows_stay_inside_one_document(tiny: Tiny) -> None:
    part = tiny.corpus.parts["train"]
    sampler = WindowSampler(part, [1], 6, 32, 5)
    rows = sampler.draw()
    assert (rows.languages == 1).all()
    for document, words in zip(rows.documents.numpy(), rows.words.numpy(), strict=True):
        span = np.asarray(part.ids[part.offsets[document] : part.offsets[document + 1]])
        assert words.astype(np.int32).tobytes() in span.tobytes()


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs MPS")
def test_expanding_on_mps_matches_the_cpu(tiny: Tiny) -> None:
    on_mps = WindowSampler(tiny.corpus.parts["train"], [0, 1], 6, 64, 0)
    on_cpu = WindowSampler(tiny.corpus.parts["train"], [0, 1], 6, 64, 0)
    mps = expander(tiny, torch.device("mps"))
    cpu = expander(tiny, torch.device("cpu"))
    for _ in range(20):
        moved = mps(on_mps.draw())
        expected = cpu(on_cpu.draw())
        assert torch.equal(moved.targets.cpu(), expected.targets)
        assert torch.equal(moved.context.cpu(), expected.context)


def edge_part(lengths: list[int], languages: list[int]) -> Encoded:
    return Encoded(
        np.repeat(np.arange(len(lengths)), lengths).astype(np.int32),
        np.concatenate([[0], np.cumsum(lengths)]),
        np.array(languages, dtype=np.int8),
    )


def test_windows_at_document_edges_never_cross_into_the_next() -> None:
    part = edge_part([4, 5, 7, 4, 6], [0, 0, 0, 0, 0])
    rows = WindowSampler(part, [0], 3, 4000, 0).draw()
    assert (rows.words.numpy() == rows.documents.numpy()[:, None]).all()


def test_languages_are_drawn_in_proportion_to_their_windows() -> None:
    part = edge_part([13, 13, 13, 13], [0, 0, 0, 1])
    rows = WindowSampler(part, [0, 1], 3, 8000, 0).draw()
    assert float((rows.languages == 0).float().mean()) == pytest.approx(0.75, abs=0.02)
