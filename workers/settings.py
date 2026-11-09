from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import TYPE_CHECKING

from data.benchmarks import Archive, BenchmarkSources, Download
from data.bilingual import DataSettings
from data.graphones import Costs, GraphoneSettings
from framework.runtime import PRECISIONS
from framework.storage import read_json
from models.conventions import Conventions
from models.graphone import (
    EQUAL,
    FEATURES,
    GATED,
    KEPT,
    LEARNED,
    MIXES,
    SPLIT,
    STREAMS,
    TABLE,
    TABLE_NAMES,
    TRIPLE,
)
from models.shared.validate import require_positive

if TYPE_CHECKING:
    from _typeshed import DataclassInstance

EXTENDS = "extends"
UNSET_LATER = {"rotation": False, "projection_rank": None}
CODE_DIMS = "code_dims"
SCALARS = (int, float, str, bool)


@dataclass(frozen=True)
class Optimiser:
    weight_decay: float
    grad_clip: float
    betas: tuple[float, float]
    eps: float
    warmup_fraction: float


@dataclass(frozen=True)
class Evaluation:
    probe_steps: int
    probe_decay: float
    neighbours: int


@dataclass(frozen=True)
class Analysis:
    late: float


@dataclass(frozen=True)
class Tables:
    source: str
    floor: float


@dataclass(frozen=True)
class Release:
    fall: float


@dataclass(frozen=True)
class Gauge:
    interval: int
    batches: int


@dataclass(frozen=True)
class Checkpoints:
    fractions: tuple[float, ...]
    arms: tuple[str, ...]


@dataclass(frozen=True)
class Arm:
    seeds: int
    projection: bool
    input_mlp: bool
    code_dims: int
    matched: bool
    receiver_scale: int
    streams: str
    mix: str
    features: str
    shuffled: bool
    hold: bool
    release_fraction: float | None
    release_from: str | None
    table: str | None
    rotation: bool
    projection_rank: int | None

    @property
    def code_width(self) -> int:
        return STREAMS[self.streams] * self.code_dims

    @property
    def ruled(self) -> bool:
        return self.hold and self.release_fraction is None and self.release_from is None


@dataclass(frozen=True)
class Config:
    data: DataSettings
    graphones: GraphoneSettings
    trunk_parameters: int
    trunk_width: int
    context_words: int
    batch_windows: int
    passes: float
    model: Conventions
    root_seed: int
    learning_rate: float
    gate_rate: float
    optimiser: Optimiser
    hold: Release
    precision: str
    log_interval: int
    snapshot_interval: int
    gauge: Gauge
    checkpoints: Checkpoints
    tables: Tables
    evaluation: Evaluation
    analysis: Analysis
    arms: dict[str, Arm]

    def code_widths(self) -> list[int]:
        return sorted({arm.code_dims for arm in self.arms.values()})

    def table_names(self) -> list[str]:
        return sorted(
            {arm.table for arm in self.arms.values() if arm.table is not None}
        )


def fits(value: object, kind: type) -> bool:
    accepted = (int, float) if kind is float else kind
    return isinstance(value, accepted) and (kind is bool or not isinstance(value, bool))


def built[T: "DataclassInstance"](
    kind: type[T], name: str, raw: dict, **nested: object
) -> T:
    expected = {item.name for item in fields(kind)}
    missing = expected - set(raw)
    unknown = set(raw) - expected
    if missing or unknown:
        raise ValueError(
            f"{name}: missing {sorted(missing)}, unknown {sorted(unknown)}"
        )
    for item in fields(kind):
        if item.type in SCALARS and not fits(raw[item.name], item.type):
            raise ValueError(f"{name}.{item.name} has the wrong type")
    return kind(**{**raw, **nested})


def benchmarks_from(raw: dict) -> BenchmarkSources:
    return built(
        BenchmarkSources,
        "data.benchmarks",
        raw,
        flores=built(Archive, "data.benchmarks.flores", raw["flores"]),
        treebanks={
            language: {
                part: built(Download, f"data.benchmarks.treebanks.{language}", source)
                for part, source in parts.items()
            }
            for language, parts in raw["treebanks"].items()
        },
        minimal_pairs={
            language: built(
                Download, f"data.benchmarks.minimal_pairs.{language}", source
            )
            for language, source in raw["minimal_pairs"].items()
        },
    )


def resolved_arm(arms: dict, name: str, seen: tuple[str, ...]) -> dict:
    if name in seen:
        raise ValueError(f"arm {name} extends itself: {' -> '.join((*seen, name))}")
    if name not in arms:
        raise ValueError(f"arm {seen[-1]} extends {name}, which is not an arm")
    arm = dict(arms[name])
    parent = arm.pop(EXTENDS, None)
    inherited = {} if parent is None else resolved_arm(arms, parent, (*seen, name))
    return {**inherited, **arm}


def direct_width(name: str, streams: int, given: dict, trunk_width: int) -> int:
    width = trunk_width // streams
    if trunk_width % streams:
        raise ValueError(f"trunk_width must divide into {streams} streams")
    if given.get(CODE_DIMS, width) != width:
        raise ValueError(
            f"arm {name}: an arm without a projection takes its code width "
            "from trunk_width"
        )
    return width


def arm_from(arms: dict, name: str, trunk_width: int) -> Arm:
    resolved = resolved_arm(arms, name, ())
    streams = STREAMS.get(resolved.get("streams", ""))
    if resolved.get("projection") is False and streams is not None:
        resolved[CODE_DIMS] = direct_width(name, streams, arms[name], trunk_width)
    arm = built(Arm, f"arm {name}", resolved)
    if arm.streams not in STREAMS:
        raise ValueError(f"arm {name}: streams must be one of {tuple(STREAMS)}")
    return arm


def parse(raw: dict) -> Config:
    data, graphones = raw["data"], raw["graphones"]
    optimiser = raw["optimiser"]
    config = built(
        Config,
        "config",
        raw,
        data=built(
            DataSettings, "data", data, benchmarks=benchmarks_from(data["benchmarks"])
        ),
        graphones=built(
            GraphoneSettings,
            "graphones",
            graphones,
            costs=built(Costs, "graphones.costs", graphones["costs"]),
        ),
        model=built(Conventions, "model", raw["model"]),
        optimiser=built(
            Optimiser, "optimiser", optimiser, betas=tuple(optimiser["betas"])
        ),
        hold=built(Release, "hold", raw["hold"]),
        gauge=built(Gauge, "gauge", raw["gauge"]),
        checkpoints=built(
            Checkpoints,
            "checkpoints",
            raw["checkpoints"],
            fractions=tuple(raw["checkpoints"]["fractions"]),
            arms=tuple(raw["checkpoints"]["arms"]),
        ),
        tables=built(Tables, "tables", raw["tables"]),
        evaluation=built(Evaluation, "evaluation", raw["evaluation"]),
        analysis=built(Analysis, "analysis", raw["analysis"]),
        arms={
            name: arm_from(raw["arms"], name, raw["trunk_width"])
            for name in raw["arms"]
        },
    )
    check(config)
    return config


def merged(base: dict, override: dict) -> dict:
    return {
        **base,
        **{
            key: merged(base[key], value)
            if isinstance(value, dict) and isinstance(base.get(key), dict)
            else value
            for key, value in override.items()
        },
    }


def load_config(path: Path) -> dict:
    settings = read_json(path)
    parent = settings.pop(EXTENDS, None)
    return (
        settings
        if parent is None
        else merged(load_config(path.parent / parent), settings)
    )


def read_config(path: Path) -> Config:
    return parse(load_config(path))


def described(arm: Arm) -> dict:
    return {
        name: value
        for name, value in asdict(arm).items()
        if name not in UNSET_LATER or value != UNSET_LATER[name]
    }


def identity(config: Config) -> dict:
    return {
        **asdict(config),
        "arms": {name: described(arm) for name, arm in config.arms.items()},
    }


def require_below_one(**values: float) -> None:
    for name, value in values.items():
        if not 0 <= value < 1:
            raise ValueError(f"{name} must be in [0, 1), {name}={value}")


def check_arm(name: str, arm: Arm, trunk_width: int) -> None:
    require_positive(
        seeds=arm.seeds, code_dims=arm.code_dims, receiver_scale=arm.receiver_scale
    )
    if arm.mix not in MIXES:
        raise ValueError(f"arm {name}: mix must be one of {MIXES}")
    if STREAMS[arm.streams] == 1 and arm.mix != EQUAL:
        raise ValueError(f"arm {name}: a single stream has nothing to mix")
    if arm.mix == KEPT and arm.streams != TRIPLE:
        raise ValueError(f"arm {name}: keeping the letters needs three streams")
    if arm.streams == SPLIT and arm.mix == TABLE:
        raise ValueError(f"arm {name}: a table mix needs three streams")
    if (arm.table is not None) != (arm.mix == TABLE):
        raise ValueError(f"arm {name}: a table belongs to a table mix and only to one")
    if arm.table is not None and arm.table not in TABLE_NAMES:
        raise ValueError(f"arm {name}: table must be one of {TABLE_NAMES}")
    if any(feature not in FEATURES for feature in arm.features):
        raise ValueError(f"arm {name}: features must be drawn from {FEATURES}")
    if arm.mix not in GATED and (arm.features != FEATURES or arm.shuffled):
        raise ValueError(f"arm {name}: only a gated mix reads features")
    if arm.hold and arm.mix == EQUAL:
        raise ValueError(f"arm {name}: an equal mix cannot be held")
    if arm.input_mlp and not arm.projection:
        raise ValueError(f"arm {name}: an input MLP replaces a projection")
    if arm.matched and arm.projection:
        raise ValueError(f"arm {name}: only an arm without a projection is matched")
    if arm.projection and arm.code_width == trunk_width:
        raise ValueError(
            f"arm {name}: a projection needs a code width other than trunk_width"
        )
    if arm.rotation and arm.mix not in GATED:
        raise ValueError(f"arm {name}: a rotation reads the gate's features")
    if arm.rotation and arm.code_width % 2:
        raise ValueError(f"arm {name}: a rotation turns coordinate pairs")
    if arm.projection_rank is not None:
        if not arm.projection or arm.input_mlp:
            raise ValueError(f"arm {name}: only a linear projection takes a rank")
        if not fits(arm.projection_rank, int) or not (
            0 < arm.projection_rank < min(arm.code_width, trunk_width)
        ):
            raise ValueError(
                f"arm {name}: projection_rank must be below the code and trunk widths"
            )


def check_data(data: DataSettings, voices: dict[str, str]) -> None:
    if len(data.languages) not in (1, 2):
        raise ValueError("data.languages must name one or two languages")
    for name, found in (
        ("data.sources", data.sources),
        ("graphones.voices", voices),
        ("data.benchmarks.flores", data.benchmarks.flores.members),
        ("data.benchmarks.treebanks", data.benchmarks.treebanks),
        ("data.benchmarks.minimal_pairs", data.benchmarks.minimal_pairs),
    ):
        missing = set(data.languages) - set(found)
        if missing:
            raise ValueError(f"{name} has nothing for {sorted(missing)}")
    require_positive(
        unit_coverage=data.unit_coverage,
        min_unit_bytes=data.min_unit_bytes,
        **data.bytes,
    )
    if data.unit_coverage > 1:
        raise ValueError("data.unit_coverage must be at most 1")


def check_optimiser(optimiser: Optimiser) -> None:
    if len(optimiser.betas) != 2:
        raise ValueError("optimiser.betas must hold two values")
    require_below_one(
        beta1=optimiser.betas[0],
        beta2=optimiser.betas[1],
        warmup_fraction=optimiser.warmup_fraction,
    )
    if optimiser.weight_decay < 0:
        raise ValueError("optimiser.weight_decay must not be negative")
    require_positive(grad_clip=optimiser.grad_clip, eps=optimiser.eps)


def check_release(name: str, arm: Arm, arms: dict[str, Arm]) -> None:
    fixed = arm.release_fraction is not None or arm.release_from is not None
    if fixed and not arm.hold:
        raise ValueError(f"arm {name}: only a held arm has a release to fix")
    if arm.release_fraction is not None and arm.release_from is not None:
        raise ValueError(f"arm {name}: a release is fixed by a fraction or an arm")
    if arm.release_fraction is not None and not 0 < arm.release_fraction < 1:
        raise ValueError(f"arm {name}: release_fraction must be in (0, 1)")
    if arm.release_from is None:
        return
    source = arms.get(arm.release_from)
    if source is None or not source.ruled or source.seeds < arm.seeds:
        raise ValueError(
            f"arm {name}: release_from must name an arm released by the rule "
            "with at least as many seeds"
        )


def check_checkpoints(checkpoints: Checkpoints, arms: dict[str, Arm]) -> None:
    fractions = checkpoints.fractions
    if any(not 0 < f < 1 for f in fractions) or list(fractions) != sorted(
        set(fractions)
    ):
        raise ValueError("checkpoints.fractions must rise strictly inside (0, 1)")
    missing = [name for name in checkpoints.arms if name not in arms]
    if missing:
        raise ValueError(f"checkpoints.arms names arms the config lacks: {missing}")


def check_tables(config: Config) -> None:
    require_positive(floor=config.tables.floor)
    source = config.arms.get(config.tables.source)
    tabled = [name for name, arm in config.arms.items() if arm.table is not None]
    if tabled and (source is None or source.mix != LEARNED or source.streams != TRIPLE):
        raise ValueError(
            f"arms {tabled} need tables.source to name a learned three-stream arm"
        )


def check(config: Config) -> None:
    require_positive(
        trunk_parameters=config.trunk_parameters,
        trunk_width=config.trunk_width,
        context_words=config.context_words,
        batch_windows=config.batch_windows,
        log_interval=config.log_interval,
        snapshot_interval=config.snapshot_interval,
        passes=config.passes,
        learning_rate=config.learning_rate,
        gate_rate=config.gate_rate,
        sample=config.graphones.sample,
        rounds=config.graphones.rounds,
        fall=config.hold.fall,
    )
    for section in (
        config.graphones.costs,
        config.model,
        config.evaluation,
        config.gauge,
    ):
        require_positive(**asdict(section))
    require_below_one(late=config.analysis.late, fall=config.hold.fall)
    if config.precision not in PRECISIONS:
        raise ValueError(f"precision must be one of {PRECISIONS}")
    check_data(config.data, config.graphones.voices)
    check_optimiser(config.optimiser)
    for name, arm in config.arms.items():
        check_arm(name, arm, config.trunk_width)
        check_release(name, arm, config.arms)
    check_checkpoints(config.checkpoints, config.arms)
    check_tables(config)
