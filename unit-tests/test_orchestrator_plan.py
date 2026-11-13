import json
from pathlib import Path

import pytest
from conftest import set_value

from framework.paths import ROOT, portable_path
from orchestrator.cli import Session
from orchestrator.context import load_context
from orchestrator.merge import merged
from pipeline.tasks import Train
from workers.layout import Layout
from workers.settings import UNSET_LATER, load_config

REVISION = "a" * 40


def session_for(tmp_path: Path, experiment: str) -> Session:
    context = load_context(
        ROOT / "config" / "smoke.json", tmp_path / "out", "selection", "cuda"
    )
    return Session([experiment], context, REVISION)


@pytest.fixture
def session(tmp_path: Path) -> Session:
    return session_for(tmp_path, "experiments/everything.py")


def labels(plan: dict) -> dict[str, str]:
    return {task["key"]: task["label"] for task in plan["tasks"]}


def parents(plan: dict, label: str) -> set[str]:
    names = labels(plan)
    task = next(t for t in plan["tasks"] if t["label"] == label)
    return {names[key] for key in task["dependencies"]}


def test_exported_plan_is_topological_and_complete(session: Session) -> None:
    plan = session.plan()
    seen = set()
    for task in plan["tasks"]:
        assert set(task["dependencies"]) <= seen
        seen.add(task["key"])
        assert task["command"][:2] == ["-m", "orchestrator"]
        assert task["command"][-2:] == ["--execute", task["key"]]
        assert task["outputs"] and not task["cached"]
    placements = {task["label"]: task["placement"] for task in plan["tasks"]}
    assert placements["Prepare"] == "local"
    assert placements["Design"] == "local"
    assert placements["Graphones"] == "local"
    assert placements["Experiment[learned-mix]"] == "local"
    assert placements["Train[learned-direct]"] == "any"
    gpu = {task["label"]: task["gpu"] for task in plan["tasks"]}
    assert gpu["Prepare"] == 0 and gpu["Train[learned-direct]"] == 1


def test_tables_wait_for_the_learned_mix_they_freeze(session: Session) -> None:
    plan = session.plan()
    assert parents(plan, "MixTables") == {"Graphones", "Train[learned-direct]"}
    assert "MixTables" in parents(plan, "Train[table-avg]")
    assert "MixTables" not in parents(plan, "Train[learned-direct]")


def test_every_experiment_reads_both_training_and_evaluation(session: Session) -> None:
    plan = session.plan()
    found = parents(plan, "Experiment[receiver-size]")
    for name in ("baseline-small", "learned-direct-large"):
        assert f"Train[{name}]" in found
        assert f"Evaluate[{name}/0/selection]" in found


def test_a_single_experiment_plans_only_its_own_arms(tmp_path: Path) -> None:
    plan = session_for(tmp_path, "experiments/learned_mix.py").plan()
    trained = {
        label.split("[")[1].split("]")[0].split("/")[0]
        for label in labels(plan).values()
        if label.startswith("Train[")
    }
    assert trained == {
        "learned-direct",
        "learned-direct-matched",
        "baseline",
        "baseline-mlp",
        "split-direct",
    }


def test_execute_refuses_a_task_whose_inputs_are_unverified(session: Session) -> None:
    plan = session.plan()
    evaluate = next(
        task
        for task in plan["tasks"]
        if task["label"] == "Evaluate[learned-direct/0/selection]"
    )
    with pytest.raises(ValueError, match="has no verified outputs here"):
        session.execute(evaluate["key"])


def paired(session: Session) -> tuple[dict, dict]:
    plan = {**session.plan(), "config": "config/en-fr.json"}
    other = {
        **plan,
        "config": "config/en-ru.json",
        "tasks": [{**task, "key": f"x{task['key']}"} for task in plan["tasks"]],
    }
    return plan, other


def test_merged_plans_keep_every_task_and_name_its_pair(session: Session) -> None:
    plan, other = paired(session)
    found = merged([plan, other], False)
    assert len(found["tasks"]) == 2 * len(plan["tasks"])
    assert found["tasks"][0]["label"] == f"en-fr/{plan['tasks'][0]['label']}"
    assert found["tasks"][-1]["label"].startswith("en-ru/")
    with pytest.raises(ValueError, match="appears in two plans"):
        merged([plan, plan], False)


def keys_for(
    root: Path, output: Path, path: tuple[str, ...], value: object
) -> dict[str, str]:
    settings = load_config(ROOT / "config" / "smoke.json")
    if path:
        set_value(path, value, settings)
    config = root / "smoke.json"
    config.write_text(json.dumps(settings))
    context = load_context(config, output, "selection", "cuda")
    plan = Session(["experiments/everything.py"], context, REVISION).plan()
    return {task["label"]: task["key"] for task in plan["tasks"]}


@pytest.fixture(scope="module")
def output(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("out")


@pytest.fixture(scope="module")
def base(tmp_path_factory: pytest.TempPathFactory, output: Path) -> dict[str, str]:
    return keys_for(tmp_path_factory.mktemp("base"), output, (), None)


def matching(keys: dict[str, str], prefix: str) -> list[str]:
    found = [
        label
        for label in keys
        if label == prefix or prefix.endswith("[") and label.startswith(prefix)
    ]
    assert found, prefix
    return found


KEY_CHANGES = [
    (
        ("arms", "baseline-large", "seeds"),
        3,
        ("Prepare", "Graphones", "MixTables", "Train[baseline]"),
        (),
    ),
    (("graphones", "costs", "pair"), 5.0, ("Prepare",), ("Graphones",)),
    (("graphones", "voices", "ru"), "ru-other", ("Graphones",), ()),
    (("evaluation", "probe_steps"), 7, ("Train[baseline]",), ("Evaluate[",)),
    (("snapshot_interval",), 7, ("Prepare",), ("Train[baseline]",)),
    (("trunk_width",), 120, ("Prepare",), ("Design", "Train[baseline]")),
    (("root_seed",), 1, ("Prepare", "Design"), ("Graphones", "Train[baseline]")),
    (
        ("arms", "spelling-only", "features"),
        "HG",
        ("Train[learned-direct]",),
        ("Train[spelling-only]",),
    ),
    (("arms", "learned", "code_dims"), 44, ("Prepare",), ("Graphones",)),
    (("arms", "table-global", "table"), "avg", (), ("MixTables",)),
]


@pytest.mark.parametrize(("path", "value", "same", "different"), KEY_CHANGES)
def test_a_task_key_changes_only_with_what_the_task_reads(
    tmp_path: Path,
    output: Path,
    base: dict[str, str],
    path: tuple[str, ...],
    value: object,
    same: tuple[str, ...],
    different: tuple[str, ...],
) -> None:
    changed = keys_for(tmp_path, output, path, value)
    assert all(
        changed[label] == base[label] for name in same for label in matching(base, name)
    )
    assert all(
        changed[label] != base[label]
        for name in different
        for label in matching(base, name)
    )


def test_more_seeds_add_tasks(
    tmp_path: Path, output: Path, base: dict[str, str]
) -> None:
    seeded = keys_for(tmp_path, output, ("arms", "baseline-large", "seeds"), 3)
    assert len(seeded) > len(base)


def test_an_experiment_refuses_a_config_without_its_arms(tmp_path: Path) -> None:
    settings = load_config(ROOT / "config" / "smoke.json")
    del settings["arms"]["split-direct"]
    config = tmp_path / "smoke.json"
    config.write_text(json.dumps(settings))
    context = load_context(config, tmp_path / "out", "selection", "cuda")
    with pytest.raises(ValueError, match=r"lacks: \['split-direct'\]"):
        Session(["experiments/learned_mix.py"], context, REVISION)


def test_an_evaluation_waits_for_its_model(session: Session) -> None:
    plan = session.plan()
    for label in labels(plan).values():
        if label.startswith("Evaluate["):
            arm = label.split("[")[1].split("/")[0]
            assert f"Train[{arm}]" in parents(plan, label)


def test_every_file_a_pod_writes_is_declared(session: Session) -> None:
    plan = session.plan()
    outputs = {task["label"]: set(task["outputs"]) for task in plan["tasks"]}
    layout = Layout(session.context.output)
    for seed in range(2):
        written = {
            layout.checkpoint("baseline", seed),
            layout.checkpoint("baseline", seed).with_suffix(".json"),
            layout.metrics("baseline", seed),
            layout.gates("baseline", seed),
            layout.steps("baseline", seed),
        }
        assert {portable_path(p) for p in written} <= outputs["Train[baseline]"]
        evaluated = {
            layout.evaluation("baseline", seed, "selection"),
            layout.scores("baseline", seed, "selection"),
        }
        assert {portable_path(p) for p in evaluated} <= outputs[
            f"Evaluate[baseline/{seed}/selection]"
        ]


def test_the_headline_and_everything_it_needs_come_first(session: Session) -> None:
    plan = session.plan()
    priority = {task["label"]: task["priority"] for task in plan["tasks"]}
    first = [
        "Prepare",
        "Design",
        "Graphones",
        "Train[learned-direct]",
        "Train[baseline]",
        "Evaluate[learned-direct/0/selection]",
        "Evaluate[baseline/1/selection]",
        "Experiment[headline]",
    ]
    assert all(priority[label] == 0 for label in first)
    assert priority["Train[triple]"] > 0 and priority["Experiment[learned-mix]"] > 0
    assert priority["Everything"] == max(priority.values())


def test_merged_plans_rank_every_stage_of_one_pair_before_the_next(
    session: Session,
) -> None:
    plan, other = paired(session)
    found = merged([plan, other], False)["tasks"]
    first = [task["priority"] for task in found if task["label"].startswith("en-fr/")]
    second = [task["priority"] for task in found if task["label"].startswith("en-ru/")]
    assert max(first) < min(second)


def test_merged_by_stage_ranks_each_stage_of_every_pair_before_the_next(
    session: Session,
) -> None:
    plan, other = paired(session)
    stages = {task["key"]: task["priority"] for task in plan["tasks"]}
    found = merged([plan, other], True)["tasks"]
    for early in found:
        for late in found:
            if (
                stages[early["key"].removeprefix("x")]
                < stages[late["key"].removeprefix("x")]
            ):
                assert early["priority"] < late["priority"]


def test_a_cuda_plan_needs_a_committed_repository(tmp_path: Path) -> None:
    for found in (f"{REVISION}-dirty", "none"):
        dirty = session_for(tmp_path, "experiments/headline.py")
        dirty.revision = found
        with pytest.raises(ValueError, match="committed repository"):
            dirty.plan()
    context = load_context(
        ROOT / "config" / "smoke.json", tmp_path / "cpu", "selection", "cpu"
    )
    plan = Session(["experiments/headline.py"], context, "none").plan()
    assert plan["revision"] == "none"


def test_every_task_carries_the_plans_revision(session: Session) -> None:
    plan = session.plan()
    assert plan["revision"] == REVISION
    for task in plan["tasks"]:
        at = task["command"].index("--revision")
        assert task["command"][at + 1] == REVISION


def test_plans_from_different_revisions_are_not_merged(session: Session) -> None:
    plan, paired_plan = paired(session)
    other = {**paired_plan, "revision": "b" * 40}
    with pytest.raises(ValueError, match="different revisions"):
        merged([plan, other], False)


def test_fields_added_later_stay_out_of_a_task_key_until_they_are_set() -> None:
    context = load_context(
        ROOT / "config" / "smoke.json",
        ROOT / "artefacts" / "smoke",
        "selection",
        "cuda",
    )
    plain = Train(context, "learned-direct").task_parameters()["spec"]
    rotated = Train(context, "rotated-direct").task_parameters()["spec"]
    ranked = Train(context, "triple-rank-matched").task_parameters()["spec"]
    assert not set(UNSET_LATER) & set(plain)
    assert set(plain) == {
        "seeds",
        "projection",
        "input_mlp",
        "code_dims",
        "matched",
        "receiver_scale",
        "streams",
        "mix",
        "features",
        "shuffled",
        "hold",
        "release_fraction",
        "release_from",
        "table",
    }
    assert rotated["rotation"] is True and "projection_rank" not in rotated
    assert ranked["projection_rank"] == 21 and "rotation" not in ranked


def test_one_language_refuses_an_arm_released_by_the_language_gauge() -> None:
    context = load_context(
        ROOT / "config" / "en.json", ROOT / "artefacts" / "en", "selection", "cuda"
    )
    assert Train(context, "triple").arm == "triple"
    assert Train(context, "rotated-direct").arm == "rotated-direct"
    assert Train(context, "rotated-held").arm == "rotated-held"
    with pytest.raises(ValueError, match="needs two languages"):
        Train(context, "learned")
