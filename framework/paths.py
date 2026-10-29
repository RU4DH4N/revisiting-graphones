from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTEFACTS = ROOT / "artefacts"


def portable_path(path: Path) -> str:
    resolved = path.resolve()
    inside = resolved.is_relative_to(ROOT)
    return (resolved.relative_to(ROOT) if inside else resolved).as_posix()
