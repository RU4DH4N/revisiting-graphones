from pathlib import Path

from framework.paths import ROOT
from framework.storage import read_json

TRUST = ROOT / "trusted.json"


def trusted(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    found: dict[str, str] = {}
    for grant in read_json(path)["grants"]:
        found.update(grant["fingerprints"])
    return found
