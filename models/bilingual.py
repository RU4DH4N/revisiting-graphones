import math

import torch
from torch import Tensor, nn

from models.graphone import GraphoneCode
from models.shared.numerics import normalise
from models.shared.symbols import BLANK, N_BYTES
from models.shared.trunk import Shape, Trunk
from models.shared.validate import require_positive

STOP = N_BYTES
START = N_BYTES + 1
RECEIVER_INPUTS = N_BYTES + 2
RECEIVER_OUTPUTS = N_BYTES + 1
PAD = BLANK
IGNORE = -100


class MessageChannel(nn.Module):
    def __init__(self, hidden: int, message: int, rank: int, std: float) -> None:
        super().__init__()
        require_positive(hidden=hidden, message=message, rank=rank, std=std)
        self.own = nn.Linear(hidden, hidden, bias=False)
        self.sent = nn.Linear(message, hidden, bias=True)
        self.down = nn.Linear(message, rank, bias=False)
        self.up = nn.Linear(rank, hidden, bias=False)
        for layer in (self.own, self.sent, self.down):
            nn.init.normal_(layer.weight, mean=0.0, std=std)
        nn.init.zeros_(self.sent.bias)
        nn.init.zeros_(self.up.weight)

    def forward(self, hidden: Tensor, message: Tensor) -> Tensor:
        gate = torch.sigmoid(self.own(hidden) + self.sent(message)[:, None, :])
        update = self.up(nn.functional.silu(self.down(message)))
        return hidden + gate * update[:, None, :]


def input_mlp(in_dims: int, shape: Shape) -> nn.Sequential:
    hidden = round(shape.mlp_ratio * shape.d_model)
    expand = nn.Linear(in_dims, hidden, bias=False)
    contract = nn.Linear(hidden, shape.d_model, bias=False)
    for layer in (expand, contract):
        nn.init.normal_(layer.weight, mean=0.0, std=shape.initialiser_range)
    return nn.Sequential(expand, nn.GELU(), contract)


class WordTrunk(nn.Module):
    code: GraphoneCode

    def __init__(
        self, shape: Shape, code: GraphoneCode, matched: bool, mlp: bool
    ) -> None:
        super().__init__()
        self.code = code
        self.trunk = Trunk(shape, code.dims, shape.message_dims)
        if mlp:
            self.trunk.in_proj = input_mlp(code.dims, shape)
        self.norm = nn.RMSNorm(shape.message_dims, eps=shape.norm_eps)
        self.eps = shape.norm_eps
        self.scale = (
            shape.initialiser_range * math.sqrt(shape.d_model) if matched else 1.0
        )

    def forward(self, units: Tensor) -> Tensor:
        codes = normalise(self.code(units), eps=self.eps) * self.scale
        return self.norm(self.trunk(codes))


class WordReceiver(nn.Module):
    def __init__(self, shape: Shape, rank: int) -> None:
        super().__init__()
        width = shape.d_model
        self.embedding = nn.Embedding(RECEIVER_INPUTS, width)
        self.body = Trunk(shape, width, width)
        self.channels = nn.ModuleList(
            MessageChannel(width, shape.message_dims, rank, shape.initialiser_range)
            for _ in range(shape.n_layers)
        )
        nn.init.normal_(self.embedding.weight, mean=0.0, std=shape.initialiser_range)

    def forward(self, inputs: Tensor, messages: Tensor) -> Tensor:
        hidden = self.embedding(inputs)
        for block, channel in zip(self.body.blocks, self.channels, strict=True):
            hidden = channel(block(hidden), messages)
        logits = nn.functional.linear(self.body.ln_f(hidden), self.embedding.weight)
        return logits[..., :RECEIVER_OUTPUTS]


def spelling_nll(logits: Tensor, targets: Tensor) -> Tensor:
    losses = nn.functional.cross_entropy(
        logits.float().flatten(0, 1),
        targets.flatten(),
        ignore_index=IGNORE,
        reduction="none",
    )
    return losses.view(targets.shape).sum(-1)


class GraphoneModel(nn.Module):
    trunk: WordTrunk
    receiver: WordReceiver

    def __init__(self, trunk: WordTrunk, receiver: WordReceiver) -> None:
        super().__init__()
        self.trunk = trunk
        self.receiver = receiver

    def messages(self, context: Tensor) -> Tensor:
        messages = self.trunk(context)
        return messages.reshape(-1, messages.shape[-1])

    def word_nll(self, inputs: Tensor, targets: Tensor, messages: Tensor) -> Tensor:
        return spelling_nll(self.receiver(inputs, messages), targets)
