import pytest
from hypothesis import given, settings, strategies as st

from data.corpus import cap_split, split_words, to_units

thorough = settings(max_examples=300, deadline=None)


def test_whitespace_attaches_to_the_word_that_follows_it() -> None:
    assert split_words("ab cd") == ["ab", " cd"]
    assert split_words("  ab  cd ") == ["  ab", "  cd", " "]
    assert split_words("​​") == ["​​"]


@pytest.mark.parametrize(
    ("unit", "max_bytes", "expected"),
    [
        ("aé", 2, ["a", "é"]),
        ("éa", 3, ["éa"]),
        ("a\U0001f60ab", 1, ["a", "\U0001f60a", "b"]),
    ],
)
def test_caps_count_encoded_bytes_and_never_break_a_character(
    unit: str, max_bytes: int, expected: list[str]
) -> None:
    assert cap_split(unit, max_bytes) == expected


def test_words_are_split_then_capped() -> None:
    assert to_units("hello  world", max_bytes=3) == ["hel", "lo", "  w", "orl", "d"]


@given(st.text(), st.integers(min_value=1, max_value=50))
@thorough
def test_segmentation_is_lossless_and_respects_the_cap(
    text: str, max_bytes: int
) -> None:
    units = to_units(text, max_bytes)
    assert "".join(units) == text
    assert "".join(split_words(text)) == text
    for unit in units:
        assert len(unit.encode("utf-8")) <= max_bytes or len(unit) == 1
