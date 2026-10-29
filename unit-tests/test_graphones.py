import math

import numpy as np
import pytest
from conftest import Tiny, config
from hypothesis import given, strategies as st
from strategies import hypothesis_settings

from data.graphones import (
    Alignment,
    align,
    align_language,
    build_units,
    core_counts,
    features,
    fold,
    has_sound,
    joint_rows,
    phones,
    refine,
    segments,
    split_key,
    stream_rows,
    type_counts,
    type_presence,
    word_parts,
)
from models.shared.symbols import BLANK
from workers.settings import parse

COSTS = parse(config()).graphones.costs

PARTS = [(" ", "", "cat", ""), (" ", "", "sat", "."), ("", "", "", ",")]
ALIGNED: Alignment = {"cat": [["k"], ["a"], ["t"]], "sat": [["s"], ["a"], ["t"]]}
SILENT: Alignment = {"cat": [["k"], ["a"], []], "sat": [["s"], ["a"], ["t"]]}


def test_alignment_gives_each_letter_its_sounds_and_loses_none() -> None:
    spoken = {"chat": phones("ʃ|a"), "cat": phones("k|a|t")}
    for word, sounds in spoken.items():
        pairs = align(word, sounds, {}, COSTS)
        assert pairs is not None and len(pairs) == len(word)
        assert [p for pair in pairs for p in pair] == sounds


def test_phones_are_espeaks_own_phonemes() -> None:
    assert phones("tʃ|ɜː|tʃ") == ["tʃ", "ɜː", "tʃ"]
    assert phones("|w|i") == ["w", "i"]
    assert phones("juː|ɛ|s eɪ") == ["juː", "ɛ", "s", "eɪ"]


def test_units_pad_with_blank_and_offset_each_language() -> None:
    units = build_units(PARTS, [ALIGNED, SILENT], 6)
    rows = len(units.keys)
    for k in range(2):
        for t in range(len(PARTS)):
            size = units.lengths[k, t]
            assert (units.codes[k, t, size:] == BLANK).all()
            inside = units.codes[k, t, :size]
            assert ((inside >= k * rows) & (inside < (k + 1) * rows)).all()
    assert units.lengths[0, 0] == 4 and units.lengths[0, 1] == 5


def test_a_long_word_is_cut_to_the_unit_width() -> None:
    units = build_units(PARTS, [ALIGNED], 3)
    assert units.lengths[0].tolist() == [3, 3, 1]


def test_letter_codes_follow_the_letter_and_sound_codes_follow_the_sound() -> None:
    keys = ["a/a", "a/ə", "b/b", "b", "t/"]
    letters, sounds = stream_rows(keys, 2, 32, 1)
    has = np.concatenate([has_sound(keys)] * 2)
    assert (letters[0] == letters[1]).all() and (letters[2] == letters[3]).all()
    assert not (sounds[0] == sounds[1]).all()
    assert has.tolist() == [True, True, True, False, True] * 2
    assert (sounds[3] == 0).all()
    assert (letters[:5] == letters[5:]).all()
    joint = joint_rows(keys, 1, 32, 2)
    assert not (joint[0] == joint[1]).all()


def test_a_split_key_keeps_a_slash_letter_whole() -> None:
    assert split_key("a/ə") == ("a", "ə")
    assert split_key("/") == ("/", None)
    assert split_key("//") == ("/", "")
    assert split_key("t/") == ("t", "")


def test_features_mark_silent_letters_and_count_uses() -> None:
    units = build_units(PARTS, [SILENT], 6)
    counts = np.array([[3, 1, 2]])
    table, uses = features(units, counts)
    silent = [units.keys.index(key) for key in units.keys if key == "t/"]
    assert table[silent, 2].tolist() == [1.0]
    assert uses[units.keys.index("a/a")] == 4
    plain = units.keys.index(",")
    assert table[plain].tolist() == [0.0, 0.0, 0.0]


def test_the_test_corpus_has_ambiguous_and_silent_letters(tiny: Tiny) -> None:
    features = tiny.tables.features
    assert (features[:, 0] > 0).any() and (features[:, 2] == 1).any()


words = st.text(alphabet="abcdeé", min_size=1, max_size=8)
phone_lists = st.lists(st.sampled_from(list("ptkaeiouə")), max_size=10)


@given(words, phone_lists)
@hypothesis_settings
def test_any_word_aligns_to_any_pronunciation_without_losing_a_sound(
    word: str, sounds: list[str]
) -> None:
    pairs = align(word, sounds, {}, COSTS)
    assert pairs is not None and len(pairs) == len(word)
    assert [p for pair in pairs for p in pair] == sounds


def test_a_two_letter_spelling_gives_its_sound_to_the_first_letter() -> None:
    cost = {("sh", "ʃ"): 0.1, ("e", "iː"): 0.1, ("ai", "ɛ"): 0.1}
    assert align("she", ["ʃ", "iː"], cost, COSTS) == [["ʃ"], [], ["iː"]]
    aid = {**cost, ("d", "d"): 0.1}
    assert align("aid", ["ɛ", "d"], aid, COSTS) == [["ɛ"], [], ["d"]]
    assert segments("she", ["ʃ", "iː"], cost, COSTS) == [("sh", ["ʃ"]), ("e", ["iː"])]


parts = st.tuples(
    st.sampled_from(["", " ", "\n"]),
    st.sampled_from(["", "(", "«"]),
    words,
    st.sampled_from(["", ".", ","]),
)


@given(st.lists(parts, min_size=1, max_size=6), st.integers(min_value=1, max_value=8))
@hypothesis_settings
def test_units_spell_out_each_word_and_pad_only_past_its_end(
    found: list[tuple[str, str, str, str]], width: int
) -> None:
    aligned: Alignment = {
        core.lower(): [[ch] for ch in core.lower()] for _, _, core, _ in found
    }
    units = build_units(found, [aligned, {}], width)
    rows = len(units.keys)
    assert rows > BLANK
    for k in range(2):
        for t, part in enumerate(found):
            size = int(units.lengths[k, t])
            assert size == min(width, len("".join(part)))
            inside = units.codes[k, t, :size] - k * rows
            assert (units.codes[k, t, size:] == BLANK).all()
            letters = "".join(split_key(units.keys[c])[0] for c in inside)
            assert letters == "".join(part)[:width]


@given(st.lists(parts, min_size=1, max_size=6))
@hypothesis_settings
def test_ambiguity_features_stay_within_their_bounds(
    found: list[tuple[str, str, str, str]],
) -> None:
    aligned: Alignment = {
        core.lower(): [[ch] for ch in core.lower()] for _, _, core, _ in found
    }
    units = build_units(found, [aligned], 8)
    table, uses = features(units, np.ones((1, len(found)), dtype=np.int64))
    sounds = len({split_key(key)[1] for key in units.keys})
    assert (table[:, :2] >= 0).all() and (table[:, :2] <= np.log2(max(sounds, 2))).all()
    assert set(np.unique(table[:, 2])) <= {0.0, 1.0}
    assert uses.sum() == units.lengths.sum()


def test_types_sharing_a_core_add_their_counts() -> None:
    parts = [(" ", "", "the", ""), ("", "", "The", ","), (" ", "", "cat", "")]
    counts = np.array([1, 5, 4])
    spoken = {"the": ["ð", "ə"], "cat": ["k", "æ", "t"]}
    assert core_counts(parts, counts, spoken) == {"the": 6, "cat": 4}


def test_folding_keeps_every_letter_in_place() -> None:
    assert fold("ÉCOLE") == "école"
    assert fold("İx") == "İx" and len("İx".lower()) == 3
    units = build_units([(" ", "", "İX", "")], [{"İx": [["i"], ["k s"]]}], 8)
    keys = [units.keys[row] for row in units.codes[0, 0, : units.lengths[0, 0]]]
    assert keys == [" ", "İ/i", "X/k s"]


def test_refined_costs_are_negative_log_shares_per_first_letter() -> None:
    said = {
        "cat": "k|a|t",
        "chat": "ʃ|a",
        "sat": "s|a|t",
        "shut": "ʃ|ʌ|t",
        "that": "ð|a|t",
    }
    spoken = {word: phones(ipa) for word, ipa in said.items()}
    cost = refine(spoken, list(spoken), 3, COSTS)
    totals: dict[str, float] = {}
    for (group, _), value in cost.items():
        assert value >= 0
        totals[group[0]] = totals.get(group[0], 0.0) + math.exp(-value)
    assert totals
    assert all(total == pytest.approx(1.0) for total in totals.values())


def test_a_sound_between_letters_joins_its_neighbour_in_order() -> None:
    cost = {("a", "a"): 0.0, ("b", "b"): 0.0, ("", "x"): 0.0}
    assert align("ab", ["a", "b", "x"], cost, COSTS) == [["a"], ["b", "x"]]
    assert align("ab", ["x", "a", "b"], cost, COSTS) == [["x", "a"], ["b"]]


def test_unit_ids_never_take_the_blank_id() -> None:
    parts = [("", "", chr(0x4E00 + k), "") for k in range(300)]
    units = build_units(parts, [{}], 4)
    assert len(np.unique(units.codes[units.codes != BLANK])) == 300
    assert units.keys[BLANK] == ""


class Listener:
    def __init__(self) -> None:
        self.heard: list[str] = []

    def phonemise(self, voice: str, words: list[str]) -> list[str]:
        self.heard.extend(words)
        return list(words)


def test_words_with_digits_are_spelled_not_spoken() -> None:
    listener = Listener()
    parts = [(" ", "", "ab1", ""), (" ", "", "ab", "")]
    aligned = align_language(
        parts, np.ones(2), np.ones(2, bool), "v", 10, 1, COSTS, listener.phonemise
    )
    assert listener.heard == ["ab"] and set(aligned) == {"ab"}


def test_type_counts_and_presence_belong_to_each_words_language(tiny: Tiny) -> None:
    cores = [part[2] for part in word_parts(tiny.corpus)]
    counts, present = type_counts(tiny.corpus), type_presence(tiny.corpus)
    for word, own in (("cat", 0), ("chat", 1)):
        ids = [t for t, core in enumerate(cores) if core == word]
        assert ids
        assert counts[own, ids].sum() > 0 and counts[1 - own, ids].sum() == 0
        assert present[own, ids].all() and not present[1 - own, ids].any()


def test_joint_codes_are_drawn_apart_from_the_stream_codes(tiny: Tiny) -> None:
    letters, sounds, joint = tiny.tables.codes_for(max(tiny.settings.code_widths()))
    streams = {
        row.tobytes() for row in np.concatenate((letters, sounds)).astype(np.int8)
    }
    assert not streams & {row.tobytes() for row in joint.astype(np.int8)}
