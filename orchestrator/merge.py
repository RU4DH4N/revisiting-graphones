import argparse
from pathlib import Path

from framework.paths import portable_path
from framework.storage import read_json, write_json
from orchestrator.contract import PLAN_FORMAT


def merged(plans: list[dict], by_stage: bool) -> dict:
    tasks = []
    seen: set[str] = set()
    span = 1 + max(task["priority"] for plan in plans for task in plan["tasks"])
    for rank, plan in enumerate(plans):
        if plan["format"] != PLAN_FORMAT:
            raise ValueError(f"plan format {plan['format']} is not {PLAN_FORMAT}")
        name = Path(plan["config"]).stem
        for task in plan["tasks"]:
            if task["key"] in seen:
                raise ValueError(f"{task['label']} appears in two plans")
            seen.add(task["key"])
            label = f"{name}/{task['label']}"
            priority = (
                task["priority"] * len(plans) + rank
                if by_stage
                else rank * span + task["priority"]
            )
            tasks.append({**task, "label": label, "priority": priority})
    revisions = {plan["revision"] for plan in plans}
    if len(revisions) != 1:
        raise ValueError(f"plans come from different revisions: {sorted(revisions)}")
    return {
        "format": PLAN_FORMAT,
        "plans": [plan["config"] for plan in plans],
        "revision": revisions.pop(),
        "tasks": tasks,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("plans", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--by-stage", action="store_true")
    args = parser.parse_args()
    plans = [read_json(path) for path in args.plans]
    write_json(args.output, merged(plans, args.by_stage))
    print(f"wrote {portable_path(args.output)}")


if __name__ == "__main__":
    main()
