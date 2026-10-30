from pathlib import Path

import numpy as np
import torch

from data.bilingual import BilingualCorpus
from data.graphones import Tables
from framework.checkpoints import load_checkpoint
from framework.seeds import seed_for
from models.bilingual import GraphoneModel, WordTrunk
from models.graphone import BYTES, FEATURES, TABLE, GraphoneCode, MixSpec, StreamTables
from models.shared.symbols import N_BYTES
from workers.batches import Expander
from workers.design import Sizing, build_receiver, receiver_for
from workers.settings import Arm, Config


def mix_spec(config: Config, arm: Arm, replicate: int) -> MixSpec:
    return MixSpec(
        streams=arm.streams,
        mix=arm.mix,
        features=arm.features,
        shuffled=arm.shuffled,
        switch=arm.hold and arm.mix == TABLE,
        gate_hidden=config.model.gate_hidden,
        gate_seed=seed_for(config.root_seed, replicate, "gate"),
        shuffle_seed=seed_for(config.root_seed, replicate, "shuffle"),
    )


def byte_tables(config: Config, arm: Arm, slots: int) -> StreamTables:
    signs = np.array([-1, 1], dtype=np.int8)
    rows = np.random.default_rng(seed_for(config.root_seed, "byte-codes")).choice(
        signs, size=(N_BYTES + 1, arm.code_dims)
    )
    empty = np.zeros_like(rows)
    return StreamTables(
        letters=rows,
        sounds=empty,
        joint=empty,
        has=np.zeros(N_BYTES + 1, dtype=bool),
        features=np.zeros((N_BYTES + 1, len(FEATURES)), dtype=np.float32),
        table=None,
        slots=slots,
    )


def expander_for(
    corpus: BilingualCorpus, tables: Tables, arm: Arm, device: torch.device
) -> Expander:
    if arm.streams != BYTES:
        return Expander(corpus, tables.codes, tables.lengths, device)
    languages = tables.codes.shape[0]
    if corpus.table.shape != tables.codes.shape[1:]:
        raise ValueError(
            f"the corpus spells words in {corpus.table.shape}, "
            f"but the graphone slots are {tables.codes.shape[1:]}"
        )
    return Expander(
        corpus,
        np.tile(corpus.table, (languages, 1, 1)),
        np.tile(corpus.lengths, (languages, 1)),
        device,
    )


def stream_tables(config: Config, tables: Tables, arm: Arm) -> StreamTables:
    if arm.streams == BYTES:
        return byte_tables(config, arm, tables.codes.shape[2])
    letters, sounds, joint = tables.codes_for(arm.code_dims)
    return StreamTables(
        letters=letters,
        sounds=sounds,
        joint=joint,
        has=tables.has,
        features=tables.features,
        table=None if arm.table is None else tables.table(arm.table),
        slots=tables.codes.shape[2],
    )


def assemble(
    config: Config, sizing: Sizing, tables: Tables, name: str, replicate: int
) -> GraphoneModel:
    arm = config.arms[name]
    torch.manual_seed(seed_for(config.root_seed, replicate, "init"))
    code = GraphoneCode(
        mix_spec(config, arm, replicate),
        stream_tables(config, tables, arm),
        seed_for(config.root_seed, replicate, "position-codes"),
    )
    trunk = WordTrunk(sizing.trunk, code, arm.matched, arm.input_mlp)
    receiver = build_receiver(
        receiver_for(config, sizing.max_unit_bytes, arm.receiver_scale), config.model
    )
    return GraphoneModel(trunk, receiver)


def load_model(
    path: Path,
    config: Config,
    sizing: Sizing,
    tables: Tables,
    name: str,
    replicate: int,
) -> GraphoneModel:
    model = assemble(config, sizing, tables, name, replicate)
    model.load_state_dict(load_checkpoint(path)["model"])
    return model
