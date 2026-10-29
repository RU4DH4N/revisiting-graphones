from pathlib import Path

from framework.paths import portable_path
from framework.storage import file_sha256
from orchestrator.contract import Contract, digest


class Graph:
    def __init__(self, roots: tuple[Contract, ...]) -> None:
        if not roots:
            raise ValueError("at least one contract is required")
        self.nodes: dict[str, Contract] = {}
        self.parents: dict[str, tuple[str, ...]] = {}
        self.fingerprints: dict[str, str] = {}
        self.order: list[str] = []
        self.visiting: list[str] = []
        self.owners: dict[Path, str] = {}
        self.hashes: dict[Path, str] = {}
        for root in roots:
            self.visit(root)

    def hashed(self, path: Path) -> str:
        if path not in self.hashes:
            self.hashes[path] = file_sha256(path)
        return self.hashes[path]

    def refuse_cycle(self, key: str) -> None:
        if key in self.visiting:
            cycle = self.visiting[self.visiting.index(key) :] + [key]
            labels = " -> ".join(self.nodes[k].label for k in cycle)
            raise ValueError(f"dependency cycle: {labels}")

    def claim(self, node: Contract, outputs: tuple[Path, ...]) -> None:
        if not outputs or len(set(outputs)) != len(outputs):
            raise ValueError(f"{node.label}: declare distinct output files")
        collisions = [output for output in outputs if output in self.owners]
        if collisions:
            raise ValueError(
                f"output collision: {collisions[0]}; use separate output "
                "directories for different configurations"
            )
        self.owners.update(dict.fromkeys(outputs, node.key))

    def add(self, node: Contract, dependencies: tuple[Contract, ...]) -> None:
        key = node.key
        parents = tuple(dict.fromkeys(dep.key for dep in dependencies))
        outputs = tuple(path.resolve() for path in node.outputs())
        sources = {portable_path(path): self.hashed(path) for path in node.sources()}
        self.claim(node, outputs)
        self.nodes[key] = node
        self.parents[key] = parents
        self.fingerprints[key] = digest(
            {
                "identity": key,
                "dependencies": parents,
                "sources": sources,
                "outputs": [portable_path(p) for p in outputs],
            }
        )

    def closure(self, key: str) -> set[str]:
        found: set[str] = set()
        pending = [key]
        while pending:
            current = pending.pop()
            if current not in found:
                found.add(current)
                pending.extend(self.parents[current])
        return found

    def stages(self, roots: tuple[Contract, ...]) -> dict[str, int]:
        branches = [dep.key for root in roots for dep in root.dependencies()]
        found: dict[str, int] = {}
        for stage, key in enumerate(branches):
            for member in self.closure(key):
                found.setdefault(member, stage)
        return {key: found.get(key, len(branches)) for key in self.order}

    def visit(self, node: Contract) -> None:
        key = node.key
        self.refuse_cycle(key)
        if key not in self.nodes:
            dependencies = node.dependencies()
            self.add(node, dependencies)
            self.visiting.append(key)
            for dependency in dependencies:
                self.visit(dependency)
            self.visiting.pop()
            self.order.append(key)
