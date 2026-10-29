import torch

from models.shared.numerics import normalise


def test_normalise_produces_unit_vectors_and_preserves_zero() -> None:
    values = torch.tensor([[3.0, 4.0], [0.0, 0.0]])
    result = normalise(values, eps=1e-6)
    torch.testing.assert_close(result[0].norm(), torch.tensor(1.0))
    torch.testing.assert_close(result[1], torch.zeros(2))
