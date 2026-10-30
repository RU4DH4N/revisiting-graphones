import argparse
from pathlib import Path

from data.splits import HALVES
from experiments.everything import BY_FILE
from framework.paths import ROOT, portable_path
from framework.revision import committed, revision
from framework.storage import write_json
from orchestrator.artefacts import Artefacts
from orchestrator.context import Context, load_context
from orchestrator.contract import PLAN_FORMAT, digest
from orchestrator.graph import Graph
from orchestrator.trust import TRUST, trusted
from pipeline.tasks import Task
from workers.entry import CONFIG, DEVICE, OUTPUT, SPLIT

EXECUTE = "--execute"
REVISION = "--revision"


def discover(stem: str) -> type[Task]:
    if stem not in BY_FILE:
        raise ValueError(f"{stem} is not an experiment; choose from {sorted(BY_FILE)}")
    return BY_FILE[stem]


class Session:
    def __init__(self, experiments: list[str], context: Context, revision: str) -> None:
        self.experiments = [Path(name).stem for name in experiments]
        self.context = context
        self.revision = revision
        roots = tuple(discover(stem)(context) for stem in self.experiments)
        self.graph = Graph(roots)
        self.stages = self.graph.stages(roots)
        self.artefacts = Artefacts(
            self.graph, context.output / ".orchestrator", trusted(TRUST)
        )

    def execute_command(self, key: str) -> list[str]:
        return [
            "-m",
            "orchestrator",
            *(f"experiments/{stem}.py" for stem in self.experiments),
            CONFIG,
            self.context.config_argument,
            OUTPUT,
            self.context.output_argument,
            SPLIT,
            self.context.split,
            DEVICE,
            self.context.device,
            REVISION,
            self.revision,
            EXECUTE,
            key,
        ]

    def plan(self) -> dict:
        if self.context.device.startswith("cuda") and not committed(self.revision):
            raise ValueError(
                f"a cuda plan needs a committed repository, found {self.revision}"
            )
        states = self.artefacts.inspect()
        tasks = []
        for key in self.graph.order:
            node = self.graph.nodes[key]
            tasks.append(
                {
                    "key": key,
                    "label": node.label,
                    "command": self.execute_command(key),
                    "dependencies": list(self.graph.parents[key]),
                    "outputs": [portable_path(path) for path in node.outputs()],
                    "manifest": portable_path(self.artefacts.path(key)),
                    "gpu": node.gpu,
                    "placement": node.placement,
                    "estimate": node.estimate(),
                    "cached": states[key][0],
                    "priority": self.stages[key],
                }
            )
        return {
            "format": PLAN_FORMAT,
            "config": self.context.config_argument,
            "output": self.context.output_argument,
            "split": self.context.split,
            "device": self.context.device,
            "revision": self.revision,
            "tasks": tasks,
        }

    def execute(self, key: str) -> None:
        if key not in self.graph.nodes:
            raise ValueError(f"no task with key {key}")
        parents = {}
        for parent in self.graph.parents[key]:
            record = self.artefacts.verified(parent)
            if record is None:
                raise ValueError(
                    f"dependency {self.graph.nodes[parent].label} "
                    "has no verified outputs here"
                )
            parents[parent] = record
        self.check_lineage(parents)
        node = self.graph.nodes[key]
        node.run()
        self.artefacts.commit(key, parents, self.revision)

    def check_lineage(self, parents: dict[str, dict]) -> None:
        for parent, record in parents.items():
            for grandparent, recorded in record["dependencies"].items():
                self.check_parent(parents, parent, grandparent, recorded)

    def check_parent(
        self, parents: dict[str, dict], parent: str, grandparent: str, recorded: str
    ) -> None:
        if (
            grandparent in parents
            and digest(parents[grandparent]["outputs"]) != recorded
        ):
            raise ValueError(
                f"{self.graph.nodes[parent].label} was built from a different "
                f"{self.graph.nodes[grandparent].label} than the one here"
            )

    def run_missing(self) -> None:
        states = self.artefacts.inspect()
        for key in (key for key in self.graph.order if not states[key][0]):
            print(f"run {self.graph.nodes[key].label}", flush=True)
            self.execute(key)


def show_plan(session: Session) -> None:
    states = session.artefacts.inspect()
    for key in session.graph.order:
        ready, reason = states[key]
        state = "cached" if ready else "pending"
        print(f"{state:>8}  {session.graph.nodes[key].label}: {reason}")


def main(arguments: list[str]) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiments", nargs="+")
    parser.add_argument(CONFIG, type=Path, required=True)
    parser.add_argument(OUTPUT, type=Path)
    parser.add_argument(SPLIT, choices=HALVES, required=True)
    parser.add_argument(DEVICE, required=True)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--plan", action="store_true")
    modes.add_argument("--run", action="store_true")
    modes.add_argument("--export", type=Path)
    modes.add_argument(EXECUTE)
    parser.add_argument(REVISION)
    args = parser.parse_args(arguments)
    output = None if args.output is None else ROOT / args.output
    context = load_context(args.config, output, args.split, args.device)
    try:
        found = revision(ROOT) if args.revision is None else args.revision
        act(Session(args.experiments, context, found), args)
    except (ValueError, FileNotFoundError) as error:
        parser.exit(1, f"{error}\n")


def act(session: Session, args: argparse.Namespace) -> None:
    if args.plan:
        show_plan(session)
    elif args.export:
        write_json(args.export, session.plan())
        print(f"wrote {portable_path(args.export)}")
    elif args.execute:
        session.execute(args.execute)
    else:
        session.run_missing()
