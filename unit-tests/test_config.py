import copy
from collections.abc import Callable
from dataclasses import asdict
from functools import partial
from pathlib import Path

import pytest
from conftest import config, set_value

from models.graphone import FEATURES
from workers.settings import parse, read_config

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
    "rotation": False,
    "projection_rank": None,
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
        ({"a": {**BASE, "streams": "letters", "mix": "learned"}}, "nothing to mix"),
        ({"a": {**BASE, "streams": "bytes", "mix": "learned"}}, "nothing to mix"),
        ({"a": {**BASE, "mix": "kept"}}, "keeping the letters needs three streams"),
        ({"a": {**BASE, "rotation": True}}, "a rotation reads the gate's features"),
        ({"a": {**BASE, "projection_rank": 0}}, "below the code and trunk widths"),
        ({"a": {**BASE, "projection_rank": 1.5}}, "below the code and trunk widths"),
        ({"a": {**BASE, "projection_rank": 64}}, "below the code and trunk widths"),
        (
            {"a": {**BASE, "input_mlp": True, "projection_rank": 4}},
            "only a linear projection takes a rank",
        ),
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
    with pytest.raises(ValueError, match="only a gated mix reads features"):
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
        (
            {"full": BASE},
            with_value(("data", "languages"), ["en", "fr", "ru"]),
            "one or two languages",
        ),
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


def test_a_config_may_name_a_single_language() -> None:
    settings = copy.deepcopy(settings_with({"full": BASE}))
    set_value(("data", "languages"), ["en"], settings)
    assert parse(settings).data.languages == ["en"]


def test_the_matched_projection_has_about_as_many_parameters_as_the_rotation() -> None:
    paper = read_config(Path(__file__).parent.parent / "config/en-fr.json")
    rotated, ranked = paper.arms["rotated-direct"], paper.arms["triple-rank-matched"]
    hidden, width = paper.model.gate_hidden, rotated.code_width
    rotation = (
        (len(FEATURES) + 4) * hidden + hidden + (hidden + 1) * (width // 2 + width)
    )
    rank = ranked.projection_rank
    assert rank is not None
    projection = rank * (ranked.code_width + paper.trunk_width)
    assert abs(projection - rotation) / rotation < 0.05
