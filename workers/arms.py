from pathlib import Path

import torch

from data.graphones import Tables
from framework.checkpoints import load_checkpoint
from framework.seeds import seed_for
from models.bilingual import GraphoneModel, WordTrunk
from models.graphone import TABLE, GraphoneCode, MixSpec, StreamTables
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


def stream_tables(tables: Tables, arm: Arm) -> StreamTables:
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
        stream_tables(tables, arm),
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
