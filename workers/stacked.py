import copy
from functools import partial

import torch
from torch import Tensor, nn
from torch.func import functional_call

from models.bilingual import (
    GraphoneModel,
    WordEncoder,
    WordReceiver,
    WordTrunk,
    spelling_nll,
)
from models.graphone import RELEASED, GraphoneCode
from models.shared.numerics import NORM_FLOOR


def stacked_parameters(modules: list[nn.Module]) -> dict[str, Tensor]:
    named = [dict(module.named_parameters()) for module in modules]
    return {
        name: torch.stack([found[name].detach() for found in named]).requires_grad_()
        for name in named[0]
    }


def per_seed_buffers(trunks: list[WordTrunk]) -> dict[str, Tensor]:
    code = trunks[0].code
    marked = {id(code.get_buffer(name)) for name in GraphoneCode.PER_SEED}
    named = [dict(trunk.named_buffers()) for trunk in trunks]
    chosen = [
        name
        for name, buffer in named[0].items()
        if id(buffer) in marked
        or any(not torch.equal(buffer, found[name]) for found in named[1:])
    ]
    return {name: torch.stack([found[name] for found in named]) for name in chosen}


def inner_names(inner: nn.Module, outer: nn.Module) -> dict[str, str]:
    outer_names = {
        id(tensor): name
        for name, tensor in (*outer.named_parameters(), *outer.named_buffers())
    }
    return {
        name: outer_names[id(tensor)]
        for name, tensor in (*inner.named_parameters(), *inner.named_buffers())
    }


def seed_messages(reader: nn.Module, own: dict[str, Tensor], inputs: Tensor) -> Tensor:
    found = functional_call(reader, own, (inputs,), strict=False)
    return found.reshape(-1, found.shape[-1])


def seed_word_nll(
    receiver: WordReceiver,
    own: dict[str, Tensor],
    inputs: Tensor,
    targets: Tensor,
    messages: Tensor,
) -> Tensor:
    logits = functional_call(receiver, own, (inputs, messages), strict=False)
    return spelling_nll(logits, targets)


class Stack:
    def __init__(self, models: list[GraphoneModel]) -> None:
        self.template = models[0]
        self.scratch = copy.deepcopy(models[0])
        trunk = self.template.trunk
        self.trunk = stacked_parameters([model.trunk for model in models])
        self.receiver = stacked_parameters([model.receiver for model in models])
        self.buffers = per_seed_buffers([model.trunk for model in models])
        self.code_names = inner_names(trunk.code, trunk)
        self.released = self.buffers[self.code_names[RELEASED]]
        self.gates = [
            self.trunk[self.code_names[name]]
            for name, _ in trunk.code.named_parameters()
        ]
        self.rotated = trunk.code.spec.rotation
        self.reader: nn.Module = WordEncoder(trunk) if self.rotated else trunk
        self.reader_names = inner_names(trunk, self.reader)

    def parameters(self) -> list[Tensor]:
        return [*self.trunk.values(), *self.receiver.values()]

    def trunk_state(self) -> dict[str, Tensor]:
        return {**self.trunk, **self.buffers}

    def messages(self, trunk: dict[str, Tensor], context: Tensor) -> Tensor:
        inputs = self.words(trunk, context) if self.rotated else context
        return self.encoded(trunk, inputs)

    def words(self, trunk: dict[str, Tensor], context: Tensor) -> Tensor:
        own = {
            inner: trunk[outer]
            for inner, outer in self.code_names.items()
            if outer in trunk
        }
        return torch.stack(
            [
                functional_call(
                    self.template.trunk.code,
                    {name: value[seed] for name, value in own.items()},
                    (rows,),
                    strict=False,
                )
                for seed, rows in enumerate(context)
            ]
        )

    def encoded(self, trunk: dict[str, Tensor], inputs: Tensor) -> Tensor:
        own = {self.reader_names[name]: value for name, value in trunk.items()}
        one = partial(seed_messages, self.reader)
        return torch.vmap(one)(own, inputs)

    def word_nll(
        self,
        receiver: dict[str, Tensor],
        inputs: Tensor,
        targets: Tensor,
        messages: Tensor,
    ) -> Tensor:
        one = partial(seed_word_nll, self.template.receiver)
        return torch.vmap(one)(receiver, inputs, targets, messages)

    def release(self, seed: int) -> None:
        self.released[seed] = True

    def model(self, seed: int) -> GraphoneModel:
        with torch.no_grad():
            for name, value in self.trunk.items():
                self.scratch.trunk.get_parameter(name).copy_(value[seed])
            for name, value in self.receiver.items():
                self.scratch.receiver.get_parameter(name).copy_(value[seed])
            for name, value in self.buffers.items():
                self.scratch.trunk.get_buffer(name).copy_(value[seed])
        return self.scratch


def clip_by_seed(grads: list[Tensor], limit: float) -> None:
    norms = torch.stack([g.float().flatten(1).square().sum(1) for g in grads])
    scale = (limit / (norms.sum(0).sqrt() + NORM_FLOOR)).clamp(max=1.0)
    for grad in grads:
        grad.mul_(scale.view(-1, *(1,) * (grad.ndim - 1)).to(grad.dtype))
