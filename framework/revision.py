import subprocess
from pathlib import Path

NO_REVISION = "none"
DIRTY = "-dirty"


def revision(root: Path) -> str:
    found = subprocess.run(
        ["git", "-C", str(root), "describe", "--always", "--dirty", "--abbrev=40"],
        capture_output=True,
        text=True,
    )
    return found.stdout.strip() if found.returncode == 0 else NO_REVISION


def committed(found: str) -> bool:
    return found != NO_REVISION and not found.endswith(DIRTY)
