"""Exact McNemar test for paired binary outcomes (SPEC §11.1, FR-REP-1)."""

from collections.abc import Sequence

from scipy.stats import binomtest  # type: ignore[import-untyped]


def mcnemar_exact_test(a: Sequence[bool], b: Sequence[bool]) -> tuple[int, int, float]:
    """Perform exact two-sided McNemar test on paired binary observations.

    Discordant pairs:
        b: a is True, b is False
        c: a is False, b is True

    Returns:
        (b_count, c_count, p_value)
    """
    if len(a) != len(b):
        raise ValueError(f"Length mismatch for paired McNemar: {len(a)} vs {len(b)}")

    b_count = 0
    c_count = 0
    for x, y in zip(a, b, strict=True):
        if x and not y:
            b_count += 1
        elif not x and y:
            c_count += 1

    total_discordant = b_count + c_count
    if total_discordant == 0:
        return (0, 0, 1.0)

    res = binomtest(min(b_count, c_count), total_discordant, p=0.5, alternative="two-sided")
    return (b_count, c_count, float(res.pvalue))
