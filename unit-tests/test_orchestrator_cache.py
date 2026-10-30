from dataclasses import dataclass
from pathlib import Path

import pytest

from framework.paths import ROOT
from framework.storage import read_json, write_json
from orchestrator.artefacts import Artefacts
from orchestrator.contract import Contract
from orchestrator.graph import Graph

REVISION = "a" * 40


@dataclass(eq=False)
class Toy(Contract):
    name: str
    files: tuple[Path, ...]
    needs: tuple[str, ...]
    code: Path
    registry: dict[str, "Toy"]

    @property
    def label(self) -> str:
        return self.name

    def parameters(self) -> dict:
        return {"name": self.name}

    def dependencies(self) -> tuple[Contract, ...]:
        return tuple(self.registry[need] for need in self.needs)

    def outputs(self) -> tuple[Path, ...]:
        return self.files

    def sources(self) -> tuple[Path, ...]:
        return (self.code,)

    def run(self) -> None:
        text = self.name + "".join(
            p.read_text() for d in self.dependencies() for p in d.outputs()
        )
        for path in self.files:
            path.write_text(text)


class Plain(Contract):
    def outputs(self) -> tuple[Path, ...]:
        return ()

    def run(self) -> None:
        return None


def toys(root: Path, needs: dict[str, tuple[str, ...]]) -> dict[str, Toy]:
    code = root / "code.py"
    if not code.exists():
        code.write_text("first")
    registry: dict[str, Toy] = {}
    for name, wanted in needs.items():
        registry[name] = Toy(name, (root / f"{name}.txt",), wanted, code, registry)
    return registry


def chain(root: Path, trust: dict[str, str]) -> tuple[Graph, Artefacts, dict[str, Toy]]:
    registry = toys(root, {"parent": (), "child": ("parent",)})
    graph = Graph((registry["child"],))
    return graph, Artefacts(graph, root / "manifests", trust), registry


def finish(graph: Graph, artefacts: Artefacts, key: str) -> None:
    parents = {}
    for parent in graph.parents[key]:
        record = artefacts.verified(parent)
        assert record is not None
        parents[parent] = record
    graph.nodes[key].run()
    artefacts.commit(key, parents, REVISION)


def ready(artefacts: Artefacts, registry: dict[str, Toy]) -> dict[str, bool]:
    states = artefacts.inspect()
    return {name: states[toy.key][0] for name, toy in registry.items()}


def finished_chain(root: Path) -> tuple[Graph, Artefacts, dict[str, Toy]]:
    graph, artefacts, registry = chain(root, {})
    for key in graph.order:
        finish(graph, artefacts, key)
    return graph, artefacts, registry


def test_finished_tasks_are_reused(tmp_path: Path) -> None:
    _, artefacts, registry = finished_chain(tmp_path)
    assert ready(artefacts, registry) == {"parent": True, "child": True}


def test_an_edited_output_reruns_the_task_and_everything_after_it(
    tmp_path: Path,
) -> None:
    _, artefacts, registry = finished_chain(tmp_path)
    registry["parent"].files[0].write_text("edited")
    assert ready(artefacts, registry) == {"parent": False, "child": False}
    assert artefacts.verified(registry["parent"].key) is None


def test_a_child_built_from_an_older_parent_is_not_reused(tmp_path: Path) -> None:
    graph, artefacts, registry = finished_chain(tmp_path)
    parent = registry["parent"]
    parent.files[0].write_text("rebuilt")
    artefacts.commit(parent.key, {}, REVISION)
    assert ready(artefacts, registry) == {"parent": True, "child": False}


def test_a_code_change_reruns_every_task_that_ran_it(tmp_path: Path) -> None:
    finished_chain(tmp_path)
    (tmp_path / "code.py").write_text("second")
    _, artefacts, registry = chain(tmp_path, {})
    assert ready(artefacts, registry) == {"parent": False, "child": False}


def regranted(root: Path) -> tuple[Graph, dict[str, str]]:
    old, _, _ = finished_chain(root)
    (root / "code.py").write_text("second")
    new, _, _ = chain(root, {})
    return old, {old.fingerprints[key]: new.fingerprints[key] for key in old.order}


def test_a_trusted_fingerprint_survives_a_code_change(tmp_path: Path) -> None:
    _, trust = regranted(tmp_path)
    _, artefacts, registry = chain(tmp_path, trust)
    assert ready(artefacts, registry) == {"parent": True, "child": True}


def test_trust_names_both_fingerprints(tmp_path: Path) -> None:
    old, _ = regranted(tmp_path)
    trust = {old.fingerprints[key]: "elsewhere" for key in old.order}
    _, artefacts, registry = chain(tmp_path, trust)
    assert ready(artefacts, registry) == {"parent": False, "child": False}


def test_trust_does_not_excuse_an_edited_output(tmp_path: Path) -> None:
    _, trust = regranted(tmp_path)
    _, artefacts, registry = chain(tmp_path, trust)
    registry["parent"].files[0].write_text("edited")
    assert ready(artefacts, registry) == {"parent": False, "child": False}


def test_a_manifest_from_another_fingerprint_is_not_trusted(tmp_path: Path) -> None:
    _, artefacts, registry = finished_chain(tmp_path)
    key = registry["parent"].key
    write_json(
        artefacts.path(key), {**read_json(artefacts.path(key)), "fingerprint": "x"}
    )
    assert artefacts.verified(key) is None


def test_a_dependency_cycle_is_refused(tmp_path: Path) -> None:
    registry = toys(tmp_path, {"a": ("b",), "b": ("a",)})
    with pytest.raises(ValueError, match="dependency cycle: a -> b -> a"):
        Graph((registry["a"],))


def test_two_tasks_cannot_write_the_same_file(tmp_path: Path) -> None:
    registry = toys(tmp_path, {"a": (), "b": ("a",)})
    registry["b"].files = registry["a"].files
    with pytest.raises(ValueError, match="output collision"):
        Graph((registry["b"],))


def test_a_task_must_name_distinct_output_files(tmp_path: Path) -> None:
    registry = toys(tmp_path, {"a": ()})
    registry["a"].files = registry["a"].files * 2
    with pytest.raises(ValueError, match="declare distinct output files"):
        Graph((registry["a"],))


def test_a_contract_hashes_the_code_of_every_contract_it_inherits() -> None:
    found = Plain().sources()
    assert ROOT / "orchestrator" / "contract.py" in found
    assert Path(__file__).resolve() in found


def test_a_manifest_records_its_revision_but_a_new_revision_still_reuses_it(
    tmp_path: Path,
) -> None:
    _, artefacts, registry = finished_chain(tmp_path)
    parent = registry["parent"].key
    assert read_json(artefacts.path(parent))["revision"] == REVISION
    artefacts.commit(parent, {}, "b" * 40)
    assert read_json(artefacts.path(parent))["revision"] == "b" * 40
    assert ready(artefacts, registry) == {"parent": True, "child": True}
