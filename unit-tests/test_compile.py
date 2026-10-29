import pytest
import torch
from conftest import Tiny, with_arm
from torch import Tensor
from torch._dynamo.utils import counters

from data.splits import TRAIN
from framework.runtime import BF16, FP32
from workers.arms import assemble
from workers.batches import Expander, SeedSamplers, WindowSampler, WordBatch
from workers.settings import Config
from workers.stacked import Stack
from workers.train import compile_for, optimisers_for, set_rate, step

CUDA = torch.device("cuda")
STEPS = 12
SEEDS = [0, 1]
ARM = "learned-direct"

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the compiled path only runs on CUDA"
)


def batches(tiny: Tiny) -> list[WordBatch]:
    expander = Expander(tiny.corpus, tiny.tables.codes, tiny.tables.lengths, CUDA)
    sampler = SeedSamplers(
        [WindowSampler(tiny.corpus.parts[TRAIN], [0, 1], 6, 4, r) for r in SEEDS]
    )
    return [expander(sampler.draw()).by_seed(len(SEEDS)) for _ in range(STEPS)]


def stacked(tiny: Tiny, settings: Config) -> Stack:
    models = [assemble(settings, tiny.sizing, tiny.tables, ARM, r) for r in SEEDS]
    return Stack([model.to(CUDA) for model in models])


def trained(
    stack: Stack, settings: Config, precision: str, data: list[WordBatch]
) -> list[Tensor]:
    optimiser, gates = optimisers_for(
        stack,
        settings.learning_rate,
        settings.gate_rate,
        settings.optimiser,
        len(SEEDS),
        CUDA,
    )
    learning = torch.ones(len(SEEDS), dtype=torch.bool, device=CUDA)
    clip = settings.optimiser.grad_clip
    found = []
    for index, batch in enumerate(data):
        set_rate(optimiser, gates, settings.learning_rate, index, STEPS, 0)
        loss, _ = step(stack, optimiser, gates, learning, batch, precision, clip, CUDA)
        found.append(loss.detach().float().clone())
    return found


def test_the_compiled_path_tracks_eager_within_the_bf16_noise(tiny: Tiny) -> None:
    settings = with_arm(tiny.settings, ARM, hold=False)
    data = batches(tiny)
    counters.clear()
    stack = stacked(tiny, settings)
    compile_for(stack, CUDA)
    addresses = {name: value.data_ptr() for name, value in stack.state().items()}
    compiled = trained(stack, settings, BF16, data)
    moved = {name: value.data_ptr() for name, value in stack.state().items()}
    eager = trained(stacked(tiny, settings), settings, BF16, data)
    exact = trained(stacked(tiny, settings), settings, FP32, data)
    assert counters["inductor"]["cudagraph_skips"] == 0
    assert moved == addresses
    torch.testing.assert_close(compiled[0], eager[0], rtol=1e-3, atol=1e-5)
    for found, plain, reference in zip(compiled, eager, exact, strict=True):
        band = 3 * (plain - reference).abs() + 1e-3 * reference.abs()
        assert torch.isfinite(found).all()
        assert ((found - plain).abs() <= band).all(), (found, plain, reference)
