import math
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.flop_counter import FlopCounterMode

from data.bilingual import BilingualCorpus
from data.graphones import Tables, read_tables
from data.splits import TRAIN
from experiments.headline import Headline
from figures.common import Inputs, written
from figures.tables import TABLES, macro
from framework.seeds import seed_for
from framework.storage import read_json
from models.bilingual import IGNORE, GraphoneModel
from workers.arms import assemble, expander_for
from workers.batches import WindowSampler, WordBatch
from workers.design import Sizing, read_sizing

NEEDS = (Headline.name,)
COMPUTE = "compute"
CPU = torch.device("cpu")


@dataclass(frozen=True)
class Cost:
    arm: str
    parameters: int
    train_step: float
    forward_word: float
    input_word: float
    gate_step: float


def counted_batch(
    inputs: Inputs, corpus: BilingualCorpus, tables: Tables, arm: str
) -> WordBatch:
    config = inputs.config
    expander = expander_for(corpus, tables, config.arms[arm], CPU)
    spoken = corpus.language_ids(config.data.languages)
    sampler = WindowSampler(
        corpus.parts[TRAIN],
        spoken,
        config.context_words,
        config.batch_windows,
        seed_for(config.root_seed, "flop-count"),
    )
    return expander(sampler.draw())


def training_flops(model: GraphoneModel, batch: WordBatch) -> float:
    with FlopCounterMode(display=False) as counter:
        nll = model.word_nll(batch.inputs, batch.targets, model.messages(batch.context))
        (nll.sum() / batch.targets.ne(IGNORE).sum()).backward()
    return float(counter.get_total_flops())


def forward_flops(model: GraphoneModel, batch: WordBatch) -> float:
    with torch.no_grad(), FlopCounterMode(display=False) as counter:
        model.word_nll(batch.inputs, batch.targets, model.messages(batch.context))
    return float(counter.get_total_flops())


def input_flops(model: GraphoneModel, batch: WordBatch) -> float:
    with torch.no_grad(), FlopCounterMode(display=False) as counter:
        model.trunk.trunk.in_proj(model.trunk.code(batch.context))
    return float(counter.get_total_flops())


def gate_flops(model: GraphoneModel) -> float:
    code = model.trunk.code
    with torch.no_grad(), FlopCounterMode(display=False) as counter:
        code.directions()
    return float(counter.get_total_flops())


def cost(
    inputs: Inputs, sizing: Sizing, tables: Tables, batch: WordBatch, arm: str
) -> Cost:
    model = assemble(inputs.config, sizing, tables, arm, 0)
    words = inputs.config.batch_windows * inputs.config.context_words
    return Cost(
        arm,
        sum(p.numel() for p in model.parameters()),
        training_flops(model, batch),
        forward_flops(model, batch) / words,
        input_flops(model, batch) / words,
        gate_flops(model),
    )


def scientific(value: float) -> str:
    if value == 0:
        return "0"
    exponent = math.floor(math.log10(abs(value)))
    return f"${value / 10**exponent:.3f} \\times 10^{{{exponent}}}$"


def row(found: Cost) -> str:
    return (
        f"{found.arm} & {found.parameters / 1e6:.2f}M & {scientific(found.train_step)} & "
        f"{scientific(found.forward_word)} & {scientific(found.input_word)} & "
        f"{scientific(found.gate_step)} \\\\"
    )


def macros(found: Cost) -> list[str]:
    name = macro(COMPUTE, found.arm)
    return [
        f"\\newcommand{{\\{name}Parameters}}{{{found.parameters / 1e6:.2f}M}}",
        f"\\newcommand{{\\{name}TrainStep}}{{{scientific(found.train_step)}}}",
        f"\\newcommand{{\\{name}ForwardWord}}{{{scientific(found.forward_word)}}}",
        f"\\newcommand{{\\{name}InputWord}}{{{scientific(found.input_word)}}}",
    ]


def compute(inputs: Inputs) -> list[Path]:
    sizing = read_sizing(read_json(inputs.layout.design))
    tables = read_tables(inputs.layout.graphones)
    corpus = BilingualCorpus(inputs.layout.corpus)
    found = [
        cost(inputs, sizing, tables, counted_batch(inputs, corpus, tables, arm), arm)
        for arm in inputs.config.arms
    ]
    folder = inputs.folder / TABLES
    lines = [line for item in found for line in macros(item)]
    return [
        written(
            folder / f"{COMPUTE}.tex", "\n".join(row(item) for item in found) + "\n"
        ),
        written(folder / f"{COMPUTE}-numbers.tex", "\n".join(lines) + "\n"),
    ]
