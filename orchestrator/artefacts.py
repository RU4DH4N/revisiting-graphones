from pathlib import Path

from framework.paths import portable_path
from framework.storage import file_sha256, read_json, write_json
from orchestrator.contract import digest
from orchestrator.graph import Graph

REVISION = "revision"
FINGERPRINT = "fingerprint"


class Artefacts:
    def __init__(self, graph: Graph, directory: Path, trust: dict[str, str]) -> None:
        self.graph, self.directory, self.trust = graph, directory, trust

    def path(self, key: str) -> Path:
        return self.directory / "cache" / f"{key}.json"

    def record(self, key: str, parents: dict[str, dict]) -> dict:
        node = self.graph.nodes[key]
        return {
            FINGERPRINT: self.graph.fingerprints[key],
            "outputs": {portable_path(p): file_sha256(p) for p in node.outputs()},
            "dependencies": {
                p: digest(parents[p]["outputs"]) for p in self.graph.parents[key]
            },
        }

    def inspect(self) -> dict[str, tuple[bool, str]]:
        states, records = {}, {}
        for key in self.graph.order:
            node = self.graph.nodes[key]
            if any(p not in records for p in self.graph.parents[key]):
                states[key] = (False, "dependency missing or stale")
                continue
            missing = [portable_path(p) for p in node.outputs() if not p.is_file()]
            if missing:
                states[key] = (False, f"missing files: {missing}")
                continue
            record = self.reusable(key, records)
            if record is not None:
                records[key] = record
                states[key] = (True, "ready")
            else:
                states[key] = (
                    False,
                    "missing manifest or changed code/configuration/inputs/outputs",
                )
        return states

    def reusable(self, key: str, parents: dict[str, dict]) -> dict | None:
        saved = self.saved(key)
        actual = self.record(key, parents)
        compared = None if saved is None else {**saved}
        if compared is not None:
            compared.pop(REVISION, None)
        return actual if actual == compared else None

    def saved(self, key: str) -> dict | None:
        try:
            found = read_json(self.path(key))
        except (OSError, ValueError):
            found = None
        if found is None:
            return None
        fingerprint = self.graph.fingerprints[key]
        recorded = found.get(FINGERPRINT, "")
        if fingerprint not in (recorded, self.trust.get(recorded)):
            return None
        return {**found, FINGERPRINT: fingerprint}

    def verified(self, key: str) -> dict | None:
        saved = self.saved(key)
        intact = saved is not None and all(
            path.is_file()
            and saved["outputs"].get(portable_path(path)) == file_sha256(path)
            for path in self.graph.nodes[key].outputs()
        )
        return saved if intact else None

    def commit(self, key: str, parents: dict[str, dict], revision: str) -> None:
        write_json(self.path(key), {**self.record(key, parents), REVISION: revision})
