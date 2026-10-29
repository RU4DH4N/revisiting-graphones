from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Layout:
    root: Path

    @property
    def corpus(self) -> Path:
        return self.root / "corpus"

    @property
    def graphones(self) -> Path:
        return self.root / "graphones"

    @property
    def design(self) -> Path:
        return self.root / "design.json"

    def run(self, arm: str, replicate: int) -> Path:
        return self.root / "arms" / arm / f"seed-{replicate}"

    def checkpoint(self, arm: str, replicate: int) -> Path:
        return self.run(arm, replicate) / "model.pt"

    def metrics(self, arm: str, replicate: int) -> Path:
        return self.run(arm, replicate) / "training.json"

    def gates(self, arm: str, replicate: int) -> Path:
        return self.run(arm, replicate) / "gates.npz"

    def steps(self, arm: str, replicate: int) -> Path:
        return self.run(arm, replicate) / "steps.npz"

    def scores(self, arm: str, replicate: int, split: str) -> Path:
        return self.run(arm, replicate) / split / "scores.npz"

    def evaluation(self, arm: str, replicate: int, split: str) -> Path:
        return self.run(arm, replicate) / split / "evaluation.json"

    def at(self, arm: str, replicate: int, fraction: float) -> Path:
        return self.run(arm, replicate) / f"at-{fraction}"

    def checkpoint_at(self, arm: str, replicate: int, fraction: float) -> Path:
        return self.at(arm, replicate, fraction) / "model.pt"

    def scores_at(self, arm: str, replicate: int, split: str, fraction: float) -> Path:
        return self.at(arm, replicate, fraction) / split / "scores.npz"

    def evaluation_at(
        self, arm: str, replicate: int, split: str, fraction: float
    ) -> Path:
        return self.at(arm, replicate, fraction) / split / "evaluation.json"

    def releases(self, arm: str) -> Path:
        return self.root / "arms" / arm / "releases.json"

    def experiment(self, name: str, split: str) -> Path:
        return self.root / "experiments" / split / f"{name}.json"
