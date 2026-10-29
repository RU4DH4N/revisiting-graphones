import json
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from conftest import Tiny

from data.graphones import table_file
from data.splits import SELECTION
from experiments.arm_names import (
    BASELINE,
    LEARNED_DIRECT,
    LEARNED_DIRECT_MATCHED,
    TABLE_AVG,
    TRIPLE_DIRECT,
)
from framework.checkpoints import load_checkpoint
from models.graphone import EQUAL, MIXED_STREAMS, TABLE_NAMES
from workers import run_evaluate, run_releases, run_tables, run_train
from workers.analysis import ArmResults, arm_report
from workers.design import sizing_record
from workers.entry import ARM, AT, CONFIG, DEVICE, OUTPUT, REPLICATE, SPLIT
from workers.layout import Layout
from workers.records import SCORES, read_gates
from workers.settings import Config, parse

FIXED_AT = "fixed-at"
ORDER = (
    BASELINE,
    TRIPLE_DIRECT,
    LEARNED_DIRECT,
    FIXED_AT,
    LEARNED_DIRECT_MATCHED,
    TABLE_AVG,
)


@pytest.fixture(scope="module")
def finished(
    tiny: Tiny, tmp_path_factory: pytest.TempPathFactory
) -> tuple[Layout, Config]:
    root = tmp_path_factory.mktemp("workers")
    shutil.copytree(tiny.root / "corpus", root / "corpus")
    shutil.copytree(tiny.root / "graphones", root / "graphones")
    sizing = sizing_record(replace(tiny.sizing, steps=6))
    (root / "design.json").write_text(json.dumps(sizing))
    arms = tiny.raw["arms"]
    raw = {
        **tiny.raw,
        "arms": {**arms, LEARNED_DIRECT: {**arms[LEARNED_DIRECT], "seeds": 2}},
        "gauge": {"interval": 1, "batches": 2},
    }
    settings = parse(raw)
    config = root / "config.json"
    config.write_text(json.dumps(raw))
    common = [CONFIG, str(config), OUTPUT, str(root), DEVICE, "cpu"]
    for name in ORDER:
        if name == TABLE_AVG:
            run_tables.main(common)
        if name == FIXED_AT:
            run_releases.main([*common, ARM, LEARNED_DIRECT])
        arm = [ARM, name]
        run_train.main([*common, *arm])
        for replicate in range(settings.arms[name].seeds):
            seed = [*common, *arm, REPLICATE, str(replicate), SPLIT, SELECTION]
            run_evaluate.main(seed)
            if name == LEARNED_DIRECT:
                run_evaluate.main([*seed, AT, "0.5"])
    return Layout(root), settings


@pytest.mark.parametrize("name", ORDER)
def test_every_kind_of_arm_trains_saves_and_scores(
    finished: tuple[Layout, Config], name: str
) -> None:
    layout, settings = finished
    checkpoint = load_checkpoint(layout.checkpoint(name, 0))
    assert checkpoint["metadata"]["arm"] == name
    metrics = json.loads(layout.metrics(name, 0).read_text())
    assert metrics["steps"] == 6 and metrics["curve"]
    evaluation = json.loads(layout.evaluation(name, 0, SELECTION).read_text())
    assert evaluation["words"] > 0
    for measure in SCORES:
        assert np.isfinite(evaluation[measure]), measure
    assert evaluation["pos_selectivity"] < evaluation["pos_accuracy"]
    gates = read_gates(layout.gates(name, 0))
    equal = settings.arms[name].mix == EQUAL
    assert (gates.final.shape == (0, MIXED_STREAMS)) == equal
    assert equal or np.allclose(np.square(gates.final).sum(1), 1.0)


def test_only_the_tables_an_arm_uses_are_built_and_each_row_is_a_direction(
    finished: tuple[Layout, Config],
) -> None:
    layout, settings = finished
    used = settings.table_names()
    assert used == ["avg"]
    for name in TABLE_NAMES:
        assert (layout.graphones / table_file(name)).exists() == (name in used)
    for name in used:
        table = np.load(layout.graphones / table_file(name))
        assert np.allclose(np.square(table).sum(1), 1.0)


def test_an_experiment_reads_what_the_workers_wrote(
    finished: tuple[Layout, Config],
) -> None:
    layout, settings = finished
    report = arm_report(ArmResults(layout, settings, LEARNED_DIRECT, SELECTION, None))
    assert report["measures"]["bits_per_byte"]["seeds"] == 2
    assert np.isfinite(report["measures"]["bits_per_byte"]["mean"])


@pytest.mark.parametrize("name", ORDER)
def test_every_seed_is_trained_and_writes_only_the_declared_files(
    finished: tuple[Layout, Config], name: str
) -> None:
    layout, settings = finished
    seeds = settings.arms[name].seeds
    assert seeds == (2 if name == LEARNED_DIRECT else 1)
    for replicate in range(seeds):
        checkpoint = layout.checkpoint(name, replicate)
        declared = {
            checkpoint,
            checkpoint.with_suffix(".json"),
            layout.metrics(name, replicate),
            layout.gates(name, replicate),
            layout.steps(name, replicate),
        }
        run = layout.run(name, replicate)
        written = {path for path in run.iterdir() if path.is_file()}
        assert written == declared


def test_a_fixed_at_arm_trains_from_its_sources_release_steps(
    finished: tuple[Layout, Config], tmp_path: Path
) -> None:
    layout, _ = finished
    root = tmp_path / "copy"
    shutil.copytree(layout.root, root)
    copied = Layout(root)
    for replicate, step in enumerate((2, 4)):
        path = copied.metrics(LEARNED_DIRECT, replicate)
        metrics = json.loads(path.read_text())
        path.write_text(json.dumps({**metrics, "released_at": step}))
    common = [CONFIG, str(root / "config.json"), OUTPUT, str(root), DEVICE, "cpu"]
    run_releases.main([*common, ARM, LEARNED_DIRECT])
    assert json.loads(copied.releases(LEARNED_DIRECT).read_text()) == [2, 4]
    run_train.main([*common, ARM, FIXED_AT])
    assert json.loads(copied.metrics(FIXED_AT, 0).read_text())["released_at"] == 2


def test_a_checkpoint_is_saved_and_scored(finished: tuple[Layout, Config]) -> None:
    layout, _ = finished
    for replicate in range(2):
        saved = load_checkpoint(layout.checkpoint_at(LEARNED_DIRECT, replicate, 0.5))
        assert saved["metadata"]["fraction"] == 0.5
        final = load_checkpoint(layout.checkpoint(LEARNED_DIRECT, replicate))
        assert any(
            not torch.equal(saved["model"][k], final["model"][k])
            for k in final["model"]
        )
        path = layout.evaluation_at(LEARNED_DIRECT, replicate, SELECTION, 0.5)
        evaluation = json.loads(path.read_text())
        assert all(np.isfinite(evaluation[measure]) for measure in SCORES)
