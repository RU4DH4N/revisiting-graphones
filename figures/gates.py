import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

from data.bilingual import BilingualCorpus
from data.graphones import Tables
from experiments.arm_names import LEARNED_DIRECT, LEARNED_SPLIT_DIRECT, SHUFFLED, TRIPLE
from experiments.mix_ablation import MixAblation
from experiments.projection_absorbs import ProjectionAbsorbs
from figures.common import Inputs, written
from figures.compute import counted_batch
from figures.tables import MISSING, TABLES, macro
from framework.storage import read_json
from models.bilingual import GraphoneModel
from models.graphone import FEATURES
from workers.arms import load_model
from workers.design import read_sizing

NEEDS = (MixAblation.name, ProjectionAbsorbs.name)
GATE_NUMBERS = "gate-numbers"
GATED = (LEARNED_DIRECT, LEARNED_SPLIT_DIRECT, SHUFFLED)
LETTER = 0
PROBE = "projection-probe"


def weighted_corr(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    mx, my = np.average(x, weights=w), np.average(y, weights=w)
    cov = np.average((x - mx) * (y - my), weights=w)
    spread = np.average((x - mx) ** 2, weights=w) * np.average((y - my) ** 2, weights=w)
    return float(cov / np.sqrt(spread)) if spread > 0 else math.nan


def shown(value: float, layout: str) -> str:
    return MISSING if math.isnan(value) else format(value, layout)


def language_rows(tables: Tables, k: int) -> np.ndarray:
    per = len(tables.keys)
    rows = np.arange(k * per, (k + 1) * per)
    return rows[tables.has[rows] & (tables.uses[rows] > 0)]


def letter_shares(inputs: Inputs, arm: str, rows: np.ndarray) -> np.ndarray:
    return np.stack(
        [
            np.square(inputs.gates(arm, r).final[rows, LETTER])
            for r in range(inputs.seeds(arm))
        ]
    )


def gate_macros(inputs: Inputs, tables: Tables, languages: list[str]) -> list[str]:
    prefix = inputs.config_path.stem
    lines = []
    for k, language in enumerate(languages):
        rows = language_rows(tables, k)
        uses = tables.uses[rows].astype(float)
        for arm in GATED:
            shares = letter_shares(inputs, arm, rows)
            name = macro(prefix, "gate", arm, language)
            mean_share = float(np.mean([np.average(s, weights=uses) for s in shares]))
            lines.append(f"\\newcommand{{\\{name}LetterShare}}{{{mean_share:.3f}}}")
            for column, feature in enumerate(FEATURES):
                found = [
                    weighted_corr(tables.features[rows, column], s, uses)
                    for s in shares
                ]
                mean, spread = float(np.mean(found)), float(np.std(found))
                lines.append(
                    f"\\newcommand{{\\{name}Corr{feature}}}{{{shown(mean, '+.2f')}}}"
                )
                lines.append(
                    f"\\newcommand{{\\{name}Corr{feature}Spread}}{{{shown(spread, '.2f')}}}"
                )
    return lines


def effective_rank(values: np.ndarray) -> float:
    share = values / values.sum()
    return float(np.exp(-(share * np.log(share)).sum()))


def projection_shares(
    model: GraphoneModel, context: torch.Tensor
) -> tuple[float, float, float]:
    trunk = model.trunk
    projection = trunk.trunk.in_proj
    if not isinstance(projection, nn.Linear):
        raise ValueError("the projection probe needs a linear in_proj")
    with torch.no_grad():
        codes = trunk.inputs(trunk.code(context))
        codes = codes.reshape(-1, codes.shape[-1])
        weight = projection.weight
        width = codes.shape[-1] // trunk.code.mixed
        streams = [slice(k * width, (k + 1) * width) for k in range(trunk.code.mixed)]
        before = np.array([codes[:, s].square().sum(-1).mean().item() for s in streams])
        after = np.array(
            [
                (codes[:, s] @ weight[:, s].T).square().sum(-1).mean().item()
                for s in streams
            ]
        )
        singular = torch.linalg.svdvals(weight).numpy() ** 2
    return (
        float(before[LETTER] / before.sum()),
        float(after[LETTER] / after.sum()),
        effective_rank(singular),
    )


def projection_macros(
    inputs: Inputs, tables: Tables, corpus: BilingualCorpus
) -> list[str]:
    sizing = read_sizing(read_json(inputs.layout.design))
    batch = counted_batch(inputs, corpus, tables, TRIPLE, PROBE)
    found = np.array(
        [
            projection_shares(
                load_model(
                    inputs.layout.checkpoint(TRIPLE, r),
                    inputs.config,
                    sizing,
                    tables,
                    TRIPLE,
                    r,
                ),
                batch.context,
            )
            for r in range(inputs.seeds(TRIPLE))
        ]
    )
    name = macro(inputs.config_path.stem, "projection", TRIPLE)
    before, after, rank = found.mean(axis=0)
    return [
        f"\\newcommand{{\\{name}LetterBefore}}{{{before:.2f}}}",
        f"\\newcommand{{\\{name}LetterAfter}}{{{after:.2f}}}",
        f"\\newcommand{{\\{name}EffectiveRank}}{{{rank:.0f}}}",
    ]


def gate_numbers(inputs: Inputs) -> list[Path]:
    tables = inputs.tables()
    corpus = BilingualCorpus(inputs.layout.corpus)
    lines = [
        *gate_macros(inputs, tables, list(corpus.languages)),
        *projection_macros(inputs, tables, corpus),
    ]
    return [
        written(inputs.folder / TABLES / f"{GATE_NUMBERS}.tex", "\n".join(lines) + "\n")
    ]
