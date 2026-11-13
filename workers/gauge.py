import math

import torch
from torch import Tensor


# https://en.wikipedia.org/wiki/Effect_size#Eta-squared_(%CE%B72)
def variance_share(values: Tensor, labels: Tensor, classes: int) -> float:
    if len(values) == 0:
        return 0.0
    if classes < 2:
        return math.nan
    centred = values - values.mean(0)
    sums = torch.zeros(classes, values.shape[1], device=values.device).index_add_(
        0, labels, centred
    )
    sizes = torch.bincount(labels, minlength=classes).float().clamp_min(1)
    between = (sums.square().sum(1) / sizes).sum() / len(values)
    total = centred.square().sum(1).mean()
    return float(between / total)


# https://en.wikipedia.org/wiki/Principal_component_analysis#Explained_variance
def direction_share(values: Tensor) -> float:
    if len(values) < 2:
        return 0.0
    if not bool(torch.isfinite(values).all()):
        return math.nan
    precise = values.cpu().double()
    centred = precise - precise.mean(0)
    spectrum = torch.linalg.eigvalsh(centred.T @ centred)
    return float(spectrum[-1] / spectrum.sum().clamp_min(1e-30))
