import sys
from dataclasses import asdict

from data.graphones import read_tables
from framework.checkpoints import save_checkpoint
from framework.storage import read_json, write_json
from workers.entry import Invocation, arm_invocation
from workers.records import save_gates, save_steps
from workers.train import train


def releases(run: Invocation, arm: str, replicates: list[int]) -> list[int | None]:
    spec = run.config.arms[arm]
    if spec.release_from is not None:
        steps = read_json(run.layout.releases(spec.release_from))
        return [steps[replicate] for replicate in replicates]
    if spec.release_fraction is not None:
        fixed = max(1, round(spec.release_fraction * run.sizing().steps))
        return [fixed for _ in replicates]
    return [None for _ in replicates]


def main(arguments: list[str]) -> None:
    task = arm_invocation(arguments)
    run = task.run
    corpus = run.corpus()
    replicates = list(range(run.config.arms[task.arm].seeds))
    found = train(
        run.config,
        run.sizing(),
        corpus,
        read_tables(run.layout.graphones),
        task.arm,
        replicates,
        releases(run, task.arm, replicates),
        run.device,
    )
    layout = run.layout
    for replicate, trained in zip(replicates, found, strict=True):
        metadata = {"arm": task.arm, "replicate": replicate, "corpus": corpus.hashes}
        save_checkpoint(layout.checkpoint(task.arm, replicate), trained.model, metadata)
        for fraction, model in trained.checkpoints.items():
            save_checkpoint(
                layout.checkpoint_at(task.arm, replicate, fraction),
                model,
                {**metadata, "fraction": fraction},
            )
        write_json(layout.metrics(task.arm, replicate), asdict(trained.metrics))
        save_gates(layout.gates(task.arm, replicate), trained.gates)
        save_steps(layout.steps(task.arm, replicate), trained.steps)


if __name__ == "__main__":
    main(sys.argv[1:])
