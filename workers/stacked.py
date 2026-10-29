import copy
from functools import partial

import torch
from torch import Tensor
from torch.func import functional_call

from models.bilingual import GraphoneModel, WordReceiver, WordTrunk, spelling_nll
from models.graphone import RELEASED, GraphoneCode
from models.shared.numerics import NORM_FLOOR

TRUNK = "trunk."
RECEIVER = "receiver."
CODE = TRUNK + "code."


def part(state: dict[str, Tensor], prefix: str) -> dict[str, Tensor]:
    return {k[len(prefix) :]: v for k, v in state.items() if k.startswith(prefix)}


def per_seed_buffers(models: list[GraphoneModel]) -> dict[str, Tensor]:
    named = [dict(model.named_buffers()) for model in models]
    return {
        CODE + name: torch.stack([found[CODE + name] for found in named])
        for name in GraphoneCode.PER_SEED
    }


def seed_messages(trunk: WordTrunk, own: dict[str, Tensor], context: Tensor) -> Tensor:
    found = functional_call(trunk, own, (context,), strict=False)
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
        named = [dict(model.named_parameters()) for model in models]
        self.params = {
            name: torch.stack(
                [found[name].detach() for found in named]
            ).requires_grad_()
            for name in named[0]
        }
        self.buffers = per_seed_buffers(models)

    def state(self) -> dict[str, Tensor]:
        return {**self.params, **self.buffers}

    def messages(self, state: dict[str, Tensor], context: Tensor) -> Tensor:
        one = partial(seed_messages, self.template.trunk)
        return torch.vmap(one)(part(state, TRUNK), context)

    def word_nll(
        self,
        state: dict[str, Tensor],
        inputs: Tensor,
        targets: Tensor,
        messages: Tensor,
    ) -> Tensor:
        one = partial(seed_word_nll, self.template.receiver)
        return torch.vmap(one)(part(state, RECEIVER), inputs, targets, messages)

    def release(self, seed: int) -> None:
        self.buffers[CODE + RELEASED][seed] = True

    def model(self, seed: int) -> GraphoneModel:
        with torch.no_grad():
            for name, value in self.params.items():
                self.scratch.get_parameter(name).copy_(value[seed])
            for name, value in self.buffers.items():
                self.scratch.get_buffer(name).copy_(value[seed])
        return self.scratch


def clip_by_seed(grads: list[Tensor], limit: float) -> None:
    norms = torch.stack([g.float().flatten(1).square().sum(1) for g in grads])
    scale = (limit / (norms.sum(0).sqrt() + NORM_FLOOR)).clamp(max=1.0)
    for grad in grads:
        grad.mul_(scale.view(-1, *(1,) * (grad.ndim - 1)).to(grad.dtype))
