from pathlib import Path

from framework.paths import ROOT, portable_path


def test_repo_paths_are_relative_and_normalised() -> None:
    assert portable_path(ROOT / "artefacts/run/../model.pt") == "artefacts/model.pt"


def test_external_path_is_not_disguised_as_a_repo_file(tmp_path: Path) -> None:
    external = tmp_path / "elsewhere" / "model.pt"
    assert portable_path(external) == external.resolve().as_posix()
