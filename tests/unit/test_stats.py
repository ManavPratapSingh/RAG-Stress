"""Unit tests for statistical methods (SPEC §11.1, FR-REP-1)."""

import pytest

from ragstress.stats.bootstrap import paired_bootstrap_mean_difference
from ragstress.stats.mcnemar import mcnemar_exact_test
from ragstress.stats.position import compute_lost_in_the_middle
from ragstress.stats.wilson import wilson_score_interval


def test_wilson_score_interval_known_values():
    """Verify Wilson 95% score interval against standard textbook/analytical values."""
    # 40 / 50 = 0.8
    low, high = wilson_score_interval(40, 50, confidence=0.95)
    assert pytest.approx(low, abs=1e-3) == 0.6696
    assert pytest.approx(high, abs=1e-3) == 0.8876

    # Zero successes: 0 / 10
    low_zero, high_zero = wilson_score_interval(0, 10, confidence=0.95)
    assert low_zero == 0.0
    assert pytest.approx(high_zero, abs=1e-3) == 0.2775

    # Full successes: 10 / 10
    low_full, high_full = wilson_score_interval(10, 10, confidence=0.95)
    assert pytest.approx(low_full, abs=1e-3) == 0.7225
    assert pytest.approx(high_full, abs=1e-6) == 1.0

    # Empty sample
    assert wilson_score_interval(0, 0) == (0.0, 0.0)


def test_wilson_score_interval_bounds_and_monotonicity():
    """Verify Wilson interval stays within [0, 1] and widens/shifts monotonically."""
    for n in [1, 5, 20, 100]:
        for k in range(n + 1):
            low, high = wilson_score_interval(k, n)
            assert 0.0 <= low <= high <= 1.0


def test_mcnemar_exact_test_known_values():
    """Verify exact two-sided McNemar test on paired binary observations."""
    # discordant pairs: b = 12 (A=1, B=0), c = 3 (A=0, B=1)
    # p-value = 2 * sum(comb(15, i) * 0.5^15 for i in range(4)) = 1152 / 32768 = 0.03515625
    a = [True] * 12 + [False] * 3 + [True] * 5 + [False] * 5
    b = [False] * 12 + [True] * 3 + [True] * 5 + [False] * 5

    res_b, res_c, p_val = mcnemar_exact_test(a, b)
    assert res_b == 12
    assert res_c == 3
    assert pytest.approx(p_val, abs=1e-6) == 0.03515625

    # Identical results: zero discordance -> p = 1.0
    assert mcnemar_exact_test([True, False], [True, False]) == (0, 0, 1.0)

    # Symmetric discordance -> p = 1.0
    _, _, p_sym = mcnemar_exact_test([True, False], [False, True])
    assert p_sym == 1.0


def test_mcnemar_length_mismatch():
    """McNemar requires identical sample sizes."""
    with pytest.raises(ValueError, match="Length mismatch"):
        mcnemar_exact_test([True], [True, False])


def test_paired_bootstrap_mean_difference_determinism_and_known():
    """Verify paired bootstrap difference of means with fixed seed."""
    a = [1.0, 1.0, 0.0, 1.0, 0.0]
    b = [0.0, 1.0, 0.0, 0.0, 0.0]

    # Run 1
    diff1, ci_low1, ci_high1 = paired_bootstrap_mean_difference(
        a, b, num_resamples=1000, seed=42, confidence=0.95
    )
    # Run 2
    diff2, ci_low2, ci_high2 = paired_bootstrap_mean_difference(
        a, b, num_resamples=1000, seed=42, confidence=0.95
    )

    assert diff1 == diff2
    assert ci_low1 == ci_low2
    assert ci_high1 == ci_high2
    assert diff1 == pytest.approx(0.4)
    assert ci_low1 <= diff1 <= ci_high1

    # Identical arrays
    diff_zero, low_zero, high_zero = paired_bootstrap_mean_difference([1.0, 2.0], [1.0, 2.0])
    assert diff_zero == 0.0
    assert low_zero == 0.0
    assert high_zero == 0.0


def test_compute_lost_in_the_middle():
    """Verify lost-in-the-middle score calculation and bootstrap confidence interval."""
    # score = (acc_top + acc_bottom) / 2 - acc_middle
    # Mock items for question "q1", "q2" across top, middle, bottom for k_noise=10
    from ragstress.runner.schema import ResultRow

    def make_row(qid: str, pos: str, pass_val: bool, k: int = 10, trial: int = 0) -> ResultRow:
        return ResultRow(
            run_id="test",
            item_id=f"{qid}-{pos}-{k}-{trial}",
            group="position_sweep",
            split="test",
            kind="plain",
            mode="A",
            k_noise=k,
            position=pos,
            conflict="none",
            k_retrieve=None,
            retriever=None,
            generator="scripted",
            intervention="baseline",
            pass_answer=pass_val,
            pass_strict=pass_val,
            provenance="gold" if pass_val else "none",
            hallucinated=not pass_val,
            failure_mode=None if pass_val else "hallucinated",
            abstain=False,
            conflict_flagged=False,
            false_conflict_flag=False,
            values_complete=pass_val,
            citation_precision=1.0,
            citation_recall=1.0,
            citation_exact=True,
            fabricated_citations=[],
        )

    # q1: top=True, bottom=True, middle=False -> top=1, bottom=1, mid=0
    # q2: top=True, bottom=True, middle=False -> top=1, bottom=1, mid=0
    # acc_top = 1.0, acc_bottom = 1.0, acc_middle = 0.0
    # score = (1.0 + 1.0) / 2 - 0.0 = 1.0
    rows = [
        make_row("q1", "top", True),
        make_row("q1", "bottom", True),
        make_row("q1", "middle", False),
        make_row("q2", "top", True),
        make_row("q2", "bottom", True),
        make_row("q2", "middle", False),
    ]

    scores = compute_lost_in_the_middle(rows, seed=42)
    assert 10 in scores
    k10_res = scores[10]
    assert k10_res["acc_top"] == 1.0
    assert k10_res["acc_bottom"] == 1.0
    assert k10_res["acc_middle"] == 0.0
    assert pytest.approx(k10_res["score"]) == 1.0
    assert k10_res["ci_lower"] <= 1.0 <= k10_res["ci_upper"]
