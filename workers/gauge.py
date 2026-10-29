import torch
from torch import Tensor


# https://en.wikipedia.org/wiki/Effect_size#Eta-squared_(%CE%B72)
def variance_share(values: Tensor, labels: Tensor, classes: int) -> float:
    if len(values) == 0:
        return 0.0
    centred = values - values.mean(0)
    sums = torch.zeros(classes, values.shape[1], device=values.device).index_add_(
        0, labels, centred
    )
    sizes = torch.bincount(labels, minlength=classes).float().clamp_min(1)
    between = (sums.square().sum(1) / sizes).sum() / len(values)
    total = centred.square().sum(1).mean()
    return float(between / total)
