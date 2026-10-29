import subprocess
from pathlib import Path

from framework.revision import NO_REVISION, committed, revision


def git(root: Path, *arguments: str) -> None:
    subprocess.run(
        ["git", "-C", str(root), *arguments], check=True, capture_output=True
    )


def test_a_revision_names_the_commit_and_marks_uncommitted_changes(
    tmp_path: Path,
) -> None:
    assert revision(tmp_path) == NO_REVISION
    git(tmp_path, "init", "-q")
    git(tmp_path, "config", "user.email", "test@example.invalid")
    git(tmp_path, "config", "user.name", "test")
    (tmp_path / "file.txt").write_text("one")
    git(tmp_path, "add", "file.txt")
    git(tmp_path, "commit", "-q", "-m", "one")
    clean = revision(tmp_path)
    assert len(clean) == 40 and committed(clean)
    (tmp_path / "file.txt").write_text("two")
    assert revision(tmp_path) == f"{clean}-dirty"
    assert not committed(revision(tmp_path))
    assert not committed(NO_REVISION)
