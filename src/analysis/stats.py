"""Bootstrap confidence intervals and the exact sign test."""

import math
import random


def mean(values: list[float]) -> float:
    """Arithmetic mean; asserts the list is not empty."""
    assert values, "mean: empty list"
    return sum(values) / len(values)


def percentile(sorted_values: list[float], fraction: float) -> float:
    """Value at a fraction (0..1) of an already-sorted list (nearest rank)."""
    assert sorted_values, "percentile: empty list"
    assert 0.0 <= fraction <= 1.0, f"percentile: fraction {fraction} must be in [0, 1]"
    index = min(len(sorted_values) - 1, max(0, int(round(fraction * (len(sorted_values) - 1)))))
    return sorted_values[index]


def bootstrap_ci(values: list[float], n_boot: int = 5000, seed: int = 0, level: float = 0.95) -> tuple[float, float]:
    """Percentile bootstrap interval for the mean of `values` (resampling the units with replacement)."""
    assert values, "bootstrap_ci: no values"
    assert n_boot >= 100, f"bootstrap_ci: n_boot={n_boot} is too small for a stable interval"
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    tail = (1.0 - level) / 2.0
    return percentile(means, tail), percentile(means, 1.0 - tail)


def sign_test(wins: int, losses: int) -> float:
    """Exact two-sided sign test p-value for paired wins vs losses (ties excluded beforehand)."""
    assert wins >= 0 and losses >= 0, f"sign_test: counts must be non-negative, got {wins}, {losses}"
    n = wins + losses
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(max(wins, losses), n + 1)) / 2 ** n
    return min(1.0, 2.0 * tail)


def paired_comparison(correct_a: list[bool], correct_b: list[bool]) -> dict[str, float]:
    """Units where only A is right vs only B is right, with the sign-test p-value."""
    assert len(correct_a) == len(correct_b), "paired_comparison: the two rules were scored on different units"
    only_a = sum(a and not b for a, b in zip(correct_a, correct_b))
    only_b = sum(b and not a for a, b in zip(correct_a, correct_b))
    return {"only_a": only_a, "only_b": only_b, "p_value": sign_test(only_a, only_b)}
