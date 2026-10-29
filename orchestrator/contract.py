import hashlib
import inspect
import json
from abc import ABC, abstractmethod
from pathlib import Path


def digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def estimate_of(category: str, flops: float, disk_bytes: int) -> dict:
    return {"category": category, "flops": float(flops), "disk_bytes": int(disk_bytes)}


PLAN_FORMAT = "task-plan-v1"
ANYWHERE = "any"
LOCAL = "local"
CPU = 0
SHARED = 1


class Contract(ABC):
    placement = ANYWHERE
    gpu = CPU

    @property
    def label(self) -> str:
        return type(self).__name__

    def parameters(self) -> dict:
        return {}

    @property
    def key(self) -> str:
        return digest(
            {
                "contract": f"{type(self).__module__}.{type(self).__qualname__}",
                "parameters": self.parameters(),
            }
        )

    def dependencies(self) -> tuple["Contract", ...]:
        return ()

    def estimate(self) -> dict:
        return estimate_of("local", 0.0, 0)

    @abstractmethod
    def outputs(self) -> tuple[Path, ...]:
        pass

    def sources(self) -> tuple[Path, ...]:
        return tuple(
            sorted(
                {
                    Path(inspect.getfile(cls)).resolve()
                    for cls in type(self).__mro__
                    if issubclass(cls, Contract) and cls is not ABC
                }
            )
        )

    @abstractmethod
    def run(self) -> None:
        pass
