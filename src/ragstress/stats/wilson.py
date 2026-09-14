"""Wilson score interval for proportions (SPEC §11.1, FR-REP-1)."""

import math

from scipy.stats import norm  # type: ignore[import-untyped]


def wilson_score_interval(k: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Calculate the two-sided Wilson score confidence interval for a proportion.

    Args:
        k: Number of successes.
        n: Total number of trials.
        confidence: Desired confidence level (default 0.95 for 95% CI).

    Returns:
        (lower_bound, upper_bound) clamped to [0.0, 1.0].
    """
    if n <= 0:
        return (0.0, 0.0)

    z = float(norm.ppf(1.0 - (1.0 - confidence) / 2.0))
    p = float(k) / float(n)
    denom = 1.0 + (z**2) / n
    center = (p + (z**2) / (2.0 * n)) / denom
    diff = (z * math.sqrt((p * (1.0 - p) / n) + ((z**2) / (4.0 * (n**2))))) / denom

    lower = 0.0 if k == 0 else max(0.0, center - diff)
    upper = 1.0 if k == n else min(1.0, center + diff)
    return (float(lower), float(upper))
