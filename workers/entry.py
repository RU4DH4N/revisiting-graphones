import argparse
from dataclasses import dataclass
from pathlib import Path

import torch

from data.bilingual import BilingualCorpus
from data.splits import HALVES
from framework.storage import read_json
from workers.design import Sizing, read_sizing
from workers.layout import Layout
from workers.settings import Config, read_config

CONFIG = "--config"
OUTPUT = "--output"
DEVICE = "--device"
ARM = "--arm"
REPLICATE = "--replicate"
SPLIT = "--split"
AT = "--at"


@dataclass(frozen=True)
class Invocation:
    config: Config
    layout: Layout
    device: torch.device

    def corpus(self) -> BilingualCorpus:
        return BilingualCorpus(self.layout.corpus)

    def sizing(self) -> Sizing:
        return read_sizing(read_json(self.layout.design))


@dataclass(frozen=True)
class ArmInvocation:
    run: Invocation
    arm: str


@dataclass(frozen=True)
class SeedInvocation:
    run: Invocation
    arm: str
    replicate: int
    split: str
    at: float | None

    def checkpoint(self) -> Path:
        layout = self.run.layout
        if self.at is None:
            return layout.checkpoint(self.arm, self.replicate)
        return layout.checkpoint_at(self.arm, self.replicate, self.at)

    def scores(self) -> Path:
        layout = self.run.layout
        if self.at is None:
            return layout.scores(self.arm, self.replicate, self.split)
        return layout.scores_at(self.arm, self.replicate, self.split, self.at)

    def evaluation(self) -> Path:
        layout = self.run.layout
        if self.at is None:
            return layout.evaluation(self.arm, self.replicate, self.split)
        return layout.evaluation_at(self.arm, self.replicate, self.split, self.at)


def parser() -> argparse.ArgumentParser:
    found = argparse.ArgumentParser()
    found.add_argument(CONFIG, type=Path, required=True)
    found.add_argument(OUTPUT, type=Path, required=True)
    found.add_argument(DEVICE, required=True)
    return found


def invocation_from(args: argparse.Namespace) -> Invocation:
    return Invocation(
        read_config(args.config), Layout(args.output), torch.device(args.device)
    )


def invocation(arguments: list[str]) -> Invocation:
    return invocation_from(parser().parse_args(arguments))


def arm_parser() -> argparse.ArgumentParser:
    found = parser()
    found.add_argument(ARM, required=True)
    return found


def checked_arm(found: argparse.ArgumentParser, run: Invocation, arm: str) -> str:
    if arm not in run.config.arms:
        found.error(f"{ARM} must name an arm in the config")
    return arm


def arm_invocation(arguments: list[str]) -> ArmInvocation:
    found = arm_parser()
    args = found.parse_args(arguments)
    run = invocation_from(args)
    return ArmInvocation(run, checked_arm(found, run, args.arm))


def seed_invocation(arguments: list[str]) -> SeedInvocation:
    found = arm_parser()
    found.add_argument(REPLICATE, type=int, required=True)
    found.add_argument(SPLIT, choices=HALVES, required=True)
    found.add_argument(AT, type=float)
    args = found.parse_args(arguments)
    run = invocation_from(args)
    arm = checked_arm(found, run, args.arm)
    if not 0 <= args.replicate < run.config.arms[arm].seeds:
        found.error(f"{REPLICATE} must name one of the arm's seeds")
    marked = run.config.checkpoints
    if args.at is not None and (
        arm not in marked.arms or args.at not in marked.fractions
    ):
        found.error(f"{AT} must name a checkpoint the arm saves")
    return SeedInvocation(run, arm, args.replicate, args.split, args.at)
