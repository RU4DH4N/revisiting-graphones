import ast

import pytest

from framework.sources import ROOT, absolute, closure


def files(module: str) -> set[str]:
    return {path.relative_to(ROOT).as_posix() for path in closure(module)}


def test_training_fingerprint_covers_training_code_only() -> None:
    training = files("workers.run_train")
    assert {
        "workers/run_train.py",
        "workers/train.py",
        "models/graphone.py",
        "data/graphones.py",
        "models/bilingual.py",
        "data/bilingual.py",
    } <= training
    assert not training & {"workers/evaluate.py", "workers/analysis.py"}


def test_evaluation_fingerprint_includes_the_model_it_scores() -> None:
    evaluation = files("workers.run_evaluate")
    assert {
        "workers/evaluate.py",
        "workers/arms.py",
        "models/graphone.py",
    } <= evaluation


def test_experiment_analysis_does_not_depend_on_training_code() -> None:
    experiments = files("workers.analysis")
    assert "workers/analysis.py" in experiments
    assert not experiments & {"workers/train.py", "models/bilingual.py"}


def test_unknown_modules_are_rejected() -> None:
    with pytest.raises(ValueError, match="not a module"):
        closure("workers.missing")


def test_relative_imports_climb_from_their_package() -> None:
    one, two = (
        ast.parse(text).body[0]
        for text in ("from .rope import x", "from ..graphone import y")
    )
    assert isinstance(one, ast.ImportFrom) and isinstance(two, ast.ImportFrom)
    assert absolute(one, "models.shared") == "models.shared.rope"
    assert absolute(two, "models.shared") == "models.graphone"
