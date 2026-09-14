"""Paired bootstrap difference of means (SPEC §11.1, FR-REP-1)."""

from collections.abc import Sequence

import numpy as np


def paired_bootstrap_mean_difference(
    a: Sequence[float],
    b: Sequence[float],
    num_resamples: int = 10000,
    seed: int = 42,
    confidence: float = 0.95,
) -> tuple[float, float, float]:
    """Calculate the paired difference of means and percentile bootstrap CI.

    Args:
        a: First sequence of values (e.g. baseline or intervention scores).
        b: Second sequence of paired values.
        num_resamples: Number of bootstrap iterations (default 10,000).
        seed: Seed for pseudo-random number generator for bit-exact determinism.
        confidence: Desired confidence level (default 0.95 for 95% CI).

    Returns:
        (mean_difference, ci_lower, ci_upper)
    """
    if len(a) != len(b):
        raise ValueError(f"Length mismatch for paired bootstrap: {len(a)} vs {len(b)}")
    if len(a) == 0:
        return (0.0, 0.0, 0.0)

    arr_a = np.asarray(a, dtype=float)
    arr_b = np.asarray(b, dtype=float)
    diffs = arr_a - arr_b
    mean_diff = float(np.mean(diffs))

    if np.all(diffs == diffs[0]):
        return (mean_diff, mean_diff, mean_diff)

    rng = np.random.default_rng(seed)
    n = len(diffs)
    # Generate bootstrap index matrix (num_resamples, n)
    indices = rng.integers(0, n, size=(num_resamples, n))
    boot_means = np.mean(diffs[indices], axis=1)

    alpha = (1.0 - confidence) / 2.0
    ci_lower = float(np.percentile(boot_means, alpha * 100.0))
    ci_upper = float(np.percentile(boot_means, (1.0 - alpha) * 100.0))
    return (mean_diff, ci_lower, ci_upper)
