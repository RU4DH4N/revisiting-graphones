import hashlib
import json
import os
import tempfile
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any, BinaryIO

CHUNK_BYTES = 1024 * 1024


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(partial(handle.read, CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write(path: Path, write: Callable[[BinaryIO], object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            write(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def write_bytes(raw: bytes, handle: BinaryIO) -> None:
    handle.write(raw)


def write_json(path: Path, payload: object) -> None:
    text = json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    atomic_write(path, partial(write_bytes, text.encode()))


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())
