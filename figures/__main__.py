import argparse
import sys
from collections.abc import Callable
from pathlib import Path

from experiments.release_timing import ReleaseTiming
from experiments.trajectory import Trajectory
from figures import (
    ambiguity,
    compute,
    contrasts,
    gates,
    losses,
    replay,
    tables,
    training,
)
from figures.common import Inputs, missing
from framework.paths import ARTEFACTS, portable_path
from workers.layout import Layout
from workers.settings import read_config

Draw = Callable[[Inputs], list[Path]]
FIGURES: dict[str, tuple[Draw, tuple[str, ...]]] = {
    training.LANGUAGE_SHARE: (training.language_share, training.NEEDS),
    training.GATE_SHARES: (training.gate_shares, training.NEEDS),
    losses.LOSS_GAP: (losses.loss_gap, losses.NEEDS),
    losses.LOSS_BY_FREQUENCY: (losses.loss_by_frequency, losses.NEEDS),
    ambiguity.GATE_AMBIGUITY: (ambiguity.gate_ambiguity, ambiguity.NEEDS),
    replay.RELEASE_REPLAY: (replay.release_replay, replay.NEEDS),
    contrasts.RELEASE_TIMING: (contrasts.release_timing, (ReleaseTiming.name,)),
    contrasts.TRAJECTORY: (contrasts.trajectory, (Trajectory.name,)),
    contrasts.SEEDS: (contrasts.seeds, ()),
    tables.TABLES: (tables.tables, ()),
    compute.COMPUTE: (compute.compute, compute.NEEDS),
    gates.GATE_NUMBERS: (gates.gate_numbers, gates.NEEDS),
}


def inputs_from(args: argparse.Namespace) -> Inputs:
    output = args.output or ARTEFACTS / args.config.stem
    return Inputs(read_config(args.config), args.config, Layout(output), args.split)


def drawn(inputs: Inputs, name: str) -> bool:
    draw, needs = FIGURES[name]
    lines = missing(inputs, needs)
    for line in lines:
        print(f"{name}: {line}")
    if lines:
        return False
    for path in draw(inputs):
        print(f"wrote {portable_path(path)}")
    return True


def main(arguments: list[str]) -> None:
    parser = argparse.ArgumentParser(prog="figures")
    parser.add_argument("names", nargs="+", choices=[*FIGURES, "all"])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split", choices=("selection", "reporting"), required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(arguments)
    names = list(FIGURES) if "all" in args.names else args.names
    inputs = inputs_from(args)
    results = [drawn(inputs, name) for name in names]
    if not all(results):
        sys.exit(1)


if __name__ == "__main__":
    main(sys.argv[1:])
