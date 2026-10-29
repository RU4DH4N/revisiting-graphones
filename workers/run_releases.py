import sys

from framework.storage import read_json, write_json
from workers.entry import arm_invocation
from workers.records import read_metrics


def main(arguments: list[str]) -> None:
    task = arm_invocation(arguments)
    layout = task.run.layout
    write_json(
        layout.releases(task.arm),
        [
            read_metrics(read_json(layout.metrics(task.arm, replicate))).released_at
            for replicate in range(task.run.config.arms[task.arm].seeds)
        ],
    )


if __name__ == "__main__":
    main(sys.argv[1:])
