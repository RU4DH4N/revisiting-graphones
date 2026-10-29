from functools import partial
from pathlib import Path

import torch

from framework.storage import atomic_write, file_sha256, read_json, write_json

CHECKPOINT_FORMAT_VERSION = 1


class CheckpointError(ValueError):
    pass


def sidecar_path(path: Path) -> Path:
    return path.with_suffix(".json")


def save_checkpoint(path: Path, model: torch.nn.Module, metadata: dict) -> None:
    payload = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "model": model.state_dict(),
        "metadata": metadata,
    }
    atomic_write(path, partial(torch.save, payload))
    write_json(
        sidecar_path(path),
        {"checkpoint_sha256": file_sha256(path), "metadata": metadata},
    )


def load_checkpoint(path: Path) -> dict:
    recorded = read_json(sidecar_path(path))
    if recorded["checkpoint_sha256"] != file_sha256(path):
        raise CheckpointError(
            f"{path} does not match the hash its sidecar recorded; "
            "re-run the task that wrote it"
        )
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload["format_version"] != CHECKPOINT_FORMAT_VERSION:
        raise CheckpointError(
            f"{path} has checkpoint format {payload['format_version']}"
        )
    return payload
