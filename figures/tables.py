import math
from pathlib import Path

from figures.common import Inputs, label, written
from figures.contrasts import EXPERIMENTS, measures_of

MISSING = "--"
FLOOR = 0.001


def number(value: float) -> str:
    if math.isnan(value):
        return MISSING
    return f"{value:+.3f}"


def probability(value: float) -> str:
    if math.isnan(value):
        return MISSING
    if value < FLOOR:
        return f"<{FLOOR}"
    return f"{value:.3f}"


def escaped(text: str) -> str:
    return text.replace("_", r"\_")


def equivalence_columns(found: dict) -> str:
    interval = f"{number(found['low'])}, {number(found['high'])}"
    return f"{interval} & margin {number(found['margin'])} & {probability(found['p'])}"


def paired_columns(test: dict, adjusted: float | None) -> str:
    interval = f"{number(test['low'])}, {number(test['high'])}"
    holm = MISSING if adjusted is None else probability(adjusted)
    return f"{interval} & {probability(test['p'])} & {holm}"


def row(contrast: str, measure: str, cell: dict, adjusted: float | None) -> str:
    test = cell["test"]
    tested = (
        equivalence_columns(cell["equivalence"])
        if "equivalence" in cell
        else paired_columns(test, adjusted)
    )
    return (
        f"{escaped(contrast)} & {label(measure)} & {test['n']} & {number(test['mean'])} & "
        f"{tested} & {probability(cell['sign_flip_p'])} & {cell['wins']}/{test['n']} \\\\"
    )


def body(report: dict) -> str:
    holm = report["family"]["holm"]
    lines = []
    for contrast, cells in report["contrasts"].items():
        for measure in measures_of(report):
            lines.append(
                row(
                    contrast, measure, cells[measure], holm.get(f"{contrast}/{measure}")
                )
            )
    return "\n".join(lines) + "\n"


def macro(*parts: str) -> str:
    words = [word for part in parts for word in part.replace("-", "_").split("_")]
    return "".join(word.capitalize() for word in words if word.isalpha())


def numbers(reports: dict[str, dict]) -> str:
    lines = []
    for experiment, report in reports.items():
        for contrast, cells in report["contrasts"].items():
            for measure in measures_of(report):
                test = cells[measure]["test"]
                name = macro(experiment, contrast, measure)
                lines.append(f"\\newcommand{{\\{name}Mean}}{{{number(test['mean'])}}}")
                lines.append(f"\\newcommand{{\\{name}Low}}{{{number(test['low'])}}}")
                lines.append(f"\\newcommand{{\\{name}High}}{{{number(test['high'])}}}")
                lines.append(f"\\newcommand{{\\{name}P}}{{{probability(test['p'])}}}")
    return "\n".join(lines) + "\n"


TABLES = "tables"


def tables(inputs: Inputs) -> list[Path]:
    present = [name for name in EXPERIMENTS if inputs.report_path(name).is_file()]
    reports = {name: inputs.report(name) for name in present}
    folder = inputs.folder / TABLES
    paths = [
        written(folder / f"{name}.tex", body(report))
        for name, report in reports.items()
    ]
    return [*paths, written(folder / "numbers.tex", numbers(reports))]
