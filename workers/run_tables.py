import sys

import numpy as np

from data.graphones import read_tables, table_file
from models.graphone import AVERAGE, GLOBAL, LETTERS_ONLY, MIXED_STREAMS, RULE
from workers.entry import invocation
from workers.records import read_gates


def on_sphere(rows: np.ndarray, floor: float) -> np.ndarray:
    norms = np.linalg.norm(rows, axis=-1, keepdims=True)
    equal = np.full(rows.shape, MIXED_STREAMS**-0.5)
    return np.where(norms >= floor, rows / np.maximum(norms, floor), equal)


def fitted_rule(
    average: np.ndarray,
    features: np.ndarray,
    uses: np.ndarray,
    rows: np.ndarray,
    floor: float,
) -> np.ndarray:
    design = np.column_stack([features, np.ones(len(features))])
    weight = np.sqrt(uses[rows])
    coefficients = np.linalg.lstsq(
        design[rows] * weight[:, None], average[rows] * weight[:, None], rcond=None
    )[0]
    return on_sphere(design @ coefficients, floor)


def main(arguments: list[str]) -> None:
    run = invocation(arguments)
    tables = read_tables(run.layout.graphones)
    source = run.config.tables.source
    floor = run.config.tables.floor
    finals = [
        read_gates(run.layout.gates(source, replicate)).final.astype(np.float64)
        for replicate in range(run.config.arms[source].seeds)
    ]
    average = np.mean(finals, axis=0)
    rows = np.flatnonzero(tables.has & (tables.uses > 0))
    built = {
        AVERAGE: on_sphere(average, floor),
        RULE: fitted_rule(
            average, tables.features.astype(np.float64), tables.uses, rows, floor
        ),
        GLOBAL: np.broadcast_to(
            on_sphere(
                np.average(average[rows], axis=0, weights=tables.uses[rows]), floor
            ),
            average.shape,
        ).copy(),
    }
    for name in run.config.table_names():
        table = built[name]
        table[~tables.has] = LETTERS_ONLY
        np.save(run.layout.graphones / table_file(name), table.astype(np.float32))


if __name__ == "__main__":
    main(sys.argv[1:])
