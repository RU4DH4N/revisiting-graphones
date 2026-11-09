from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

from framework.paths import ARTEFACTS, portable_path
from workers.settings import Config, identity, read_config


@dataclass(frozen=True)
class Context:
    config_path: Path
    config: Config
    output: Path
    split: str
    device: str

    @cached_property
    def identity(self) -> dict:
        return identity(self.config)

    @property
    def config_argument(self) -> str:
        return portable_path(self.config_path)

    @property
    def output_argument(self) -> str:
        return portable_path(self.output)


def load_context(config: Path, output: Path | None, split: str, device: str) -> Context:
    resolved = (output or ARTEFACTS / config.stem).resolve()
    return Context(config.resolve(), read_config(config), resolved, split, device)
