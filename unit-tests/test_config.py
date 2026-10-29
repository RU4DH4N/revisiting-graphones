import copy
from collections.abc import Callable
from dataclasses import asdict
from functools import partial

import pytest
from conftest import config, set_value

from workers.settings import parse

BASE = {
    "seeds": 2,
    "projection": True,
    "input_mlp": False,
    "code_dims": 32,
    "matched": False,
    "receiver_scale": 1,
    "streams": "split",
    "mix": "equal",
    "features": "HGS",
    "shuffled": False,
    "hold": False,
    "release_fraction": None,
    "release_from": None,
    "table": None,
}


def settings_with(arms: dict) -> dict:
    return {
        **config(),
        "arms": arms,
        "checkpoints": {"fractions": [0.5], "arms": []},
        "tables": {"source": "full", "floor": 0.01},
    }


def test_an_arm_takes_its_parent_and_changes_only_what_it_names() -> None:
    arms = parse(
        settings_with(
            {
                "base": BASE,
                "triple": {"extends": "base", "streams": "triple"},
                "full": {"extends": "triple", "mix": "learned", "hold": True},
            }
        )
    ).arms
    assert asdict(arms["full"]) == {
        **BASE,
        "streams": "triple",
        "mix": "learned",
        "hold": True,
    }
    assert asdict(arms["base"]) == BASE


def test_an_arm_without_a_projection_takes_its_width_from_the_trunk() -> None:
    settings = settings_with(
        {
            "base": BASE,
            "split": {"extends": "base", "projection": False},
            "triple": {"extends": "split", "streams": "triple"},
        }
    )
    arms = parse(settings).arms
    assert arms["split"].code_dims == settings["trunk_width"] // 2
    assert arms["triple"].code_dims == settings["trunk_width"] // 3
    stated = copy.deepcopy(settings)
    stated["arms"]["split"]["code_dims"] = settings["trunk_width"] // 2
    assert parse(stated) == parse(settings)


@pytest.mark.parametrize(
    ("arms", "message"),
    [
        ({"a": {**BASE, "extends": "b"}, "b": {"extends": "a"}}, "extends itself"),
        ({"a": {"extends": "missing"}}, "not an arm"),
        ({"a": {**BASE, "projection": False}}, "from trunk_width"),
    ],
)
def test_broken_arm_definitions_are_refused(arms: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        parse(settings_with(arms))


def test_a_projection_must_change_the_width() -> None:
    settings = settings_with({"full": {**BASE, "code_dims": 24}})
    with pytest.raises(ValueError, match="other than trunk_width"):
        parse(settings)


@pytest.mark.parametrize("changes", [{"features": "H"}, {"shuffled": True}])
def test_only_a_learned_mix_reads_features(changes: dict) -> None:
    settings = settings_with({"full": {**BASE, **changes}})
    with pytest.raises(ValueError, match="only a learned mix reads features"):
        parse(settings)


def without_voice(settings: dict) -> None:
    del settings["graphones"]["voices"]["fr"]


def with_value(path: tuple[str, ...], value: object) -> Callable[[dict], None]:
    return partial(set_value, path, value)


TABLED = {**BASE, "streams": "triple", "mix": "table", "table": "avg"}


@pytest.mark.parametrize(
    ("arms", "change", "message"),
    [
        ({"full": {**BASE, "table": "avg"}}, None, "a table belongs to a table mix"),
        ({"full": {**BASE, "colour": 1}}, None, r"unknown \['colour'\]"),
        ({"full": TABLED}, None, "need tables.source to name a learned three-stream"),
        (
            {"full": {**BASE, "mix": "learned"}, "tabled": TABLED},
            None,
            "need tables.source to name a learned three-stream",
        ),
        ({"full": BASE}, with_value(("data", "languages"), ["en"]), "exactly two"),
        ({"full": BASE}, without_voice, r"graphones.voices has nothing for \['fr'\]"),
        ({"full": BASE}, with_value(("data", "unit_coverage"), 1.5), "at most 1"),
        (
            {"full": BASE},
            with_value(("optimiser", "weight_decay"), -0.1),
            "must not be negative",
        ),
        ({"full": BASE}, with_value(("optimiser", "betas"), [0.9, 1.0]), r"\[0, 1\)"),
        (
            {"full": BASE},
            with_value(("hold", "fall"), 1.0),
            r"fall must be in \[0, 1\)",
        ),
        ({"full": BASE}, with_value(("hold", "fall"), 0.0), "fall must be positive"),
    ],
)
def test_broken_settings_are_refused(
    arms: dict, change: Callable[[dict], None] | None, message: str
) -> None:
    settings = copy.deepcopy(settings_with(arms))
    if change is not None:
        change(settings)
    with pytest.raises(ValueError, match=message):
        parse(settings)
