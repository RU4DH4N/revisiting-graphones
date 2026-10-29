import itertools
import math
from dataclasses import dataclass

import numpy as np
from scipy import stats

from workers.records import BITS_PER_BYTE

LOWER_IS_BETTER = (BITS_PER_BYTE,)
CONFIDENCE = 0.95
ALPHA = 0.05


def direction(measure: str) -> float:
    return -1.0 if measure in LOWER_IS_BETTER else 1.0


@dataclass(frozen=True)
class Test:
    n: int
    mean: float
    sd: float
    low: float
    high: float
    p: float


@dataclass(frozen=True)
class Cell:
    measure: str
    replicates: int
    failed_left: int
    failed_right: int
    left: list[float]
    right: list[float]
    differences: list[float]
    test: Test
    sensitivity: Test
    sign_flip_p: float
    sign_flip_floor: float
    wins: int
    d_z: float


def nothing() -> Test:
    return Test(0, math.nan, math.nan, math.nan, math.nan, math.nan)


# https://en.wikipedia.org/wiki/Student%27s_t-test#Dependent_t-test_for_paired_samples
def paired_t(differences: np.ndarray) -> Test:
    n = len(differences)
    if n < 2:
        return nothing()
    mean, sd = float(differences.mean()), float(differences.std(ddof=1))
    if sd == 0:
        return Test(n, mean, sd, mean, mean, 1.0 if mean == 0 else 0.0)
    error = sd / math.sqrt(n)
    reach = float(stats.t.ppf(0.5 + CONFIDENCE / 2, n - 1)) * error
    p = float(2 * stats.t.sf(abs(mean / error), n - 1))
    return Test(n, mean, sd, mean - reach, mean + reach, p)


# https://en.wikipedia.org/wiki/Permutation_test
def sign_flip_p(differences: np.ndarray) -> float:
    n = len(differences)
    if n == 0:
        return math.nan
    observed = abs(differences.mean())
    signs = np.array(list(itertools.product((-1.0, 1.0), repeat=n)))
    flipped = np.abs((signs * differences).mean(axis=1))
    return float(np.mean(flipped >= observed - 1e-12))


def imputed(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    worst_left = np.nanmin(left) if np.isfinite(left).any() else math.nan
    best_right = np.nanmax(right) if np.isfinite(right).any() else math.nan
    filled_left = np.where(np.isfinite(left), left, worst_left)
    filled_right = np.where(np.isfinite(right), right, best_right)
    return filled_left - filled_right


def paired(measure: str, left: list[float], right: list[float]) -> Cell:
    shared = min(len(left), len(right))
    sign = direction(measure)
    a = sign * np.asarray(left[:shared], dtype=float)
    b = sign * np.asarray(right[:shared], dtype=float)
    kept = np.isfinite(a) & np.isfinite(b)
    differences = a[kept] - b[kept]
    return Cell(
        measure,
        shared,
        int((~np.isfinite(a)).sum()),
        int((~np.isfinite(b)).sum()),
        left[:shared],
        right[:shared],
        differences.tolist(),
        paired_t(differences),
        paired_t(imputed(a, b)),
        sign_flip_p(differences),
        2 / 2 ** len(differences) if len(differences) else math.nan,
        int((differences > 0).sum()),
        float(differences.mean() / differences.std(ddof=1))
        if len(differences) > 1 and differences.std(ddof=1) > 0
        else math.nan,
    )


@dataclass(frozen=True)
class Equivalence:
    margin: float
    low: float
    high: float
    p: float


# https://en.wikipedia.org/wiki/Equivalence_test
def equivalence(cell: Cell) -> Equivalence:
    right = np.asarray(cell.right, dtype=float)
    right = right[np.isfinite(right)]
    differences = np.asarray(cell.differences, dtype=float)
    n = len(differences)
    if n < 2 or len(right) < 2:
        return Equivalence(math.nan, math.nan, math.nan, math.nan)
    margin = float(right.std(ddof=1))
    mean, error = (
        float(differences.mean()),
        float(differences.std(ddof=1)) / math.sqrt(n),
    )
    if error == 0:
        inside = abs(mean) < margin
        return Equivalence(margin, mean, mean, 0.0 if inside else 1.0)
    reach = float(stats.t.ppf(1 - ALPHA, n - 1)) * error
    above = float(stats.t.sf((mean + margin) / error, n - 1))
    below = float(stats.t.cdf((mean - margin) / error, n - 1))
    return Equivalence(margin, mean - reach, mean + reach, max(above, below))


# https://en.wikipedia.org/wiki/Holm%E2%80%93Bonferroni_method
def holm(found: dict[str, float]) -> dict[str, float]:
    ordered = sorted((p, name) for name, p in found.items() if not math.isnan(p))
    adjusted: dict[str, float] = {name: math.nan for name in found}
    running = 0.0
    for rank, (p, name) in enumerate(ordered):
        running = max(running, min(1.0, (len(ordered) - rank) * p))
        adjusted[name] = running
    return adjusted
