"""Position sweep and lost-in-the-middle statistics (SPEC §11.1, FR-REP-1)."""

import re
from collections.abc import Sequence
from typing import Any

import numpy as np

from ragstress.runner.schema import ResultRow


def extract_question_id(item_id: str) -> str:
    """Extract question identifier from position sweep item_id."""
    return re.sub(r"-(top|middle|bottom).*$", "", item_id)


def compute_lost_in_the_middle(
    rows: Sequence[ResultRow],
    num_resamples: int = 10000,
    seed: int = 42,
    confidence: float = 0.95,
) -> dict[int, dict[str, Any]]:
    """Compute lost-in-the-middle score and question-clustered bootstrap CI per k_noise.

    Formula:
        score = mean(acc_top, acc_bottom) - acc_middle

    Returns:
        Mapping from k_noise to {
            "acc_top": float,
            "acc_bottom": float,
            "acc_middle": float,
            "score": float,
            "ci_lower": float,
            "ci_upper": float,
            "count_top": int,
            "count_middle": int,
            "count_bottom": int,
        }
    """
    pos_rows = [r for r in rows if r.group == "position_sweep"]
    k_levels = sorted({r.k_noise for r in pos_rows})
    results: dict[int, dict[str, Any]] = {}

    for k in k_levels:
        k_rows = [r for r in pos_rows if r.k_noise == k]
        top_rows = [r for r in k_rows if r.position == "top"]
        mid_rows = [r for r in k_rows if r.position == "middle"]
        bot_rows = [r for r in k_rows if r.position == "bottom"]

        acc_top = (
            float(sum(1 for r in top_rows if r.pass_answer)) / len(top_rows) if top_rows else 0.0
        )
        acc_mid = (
            float(sum(1 for r in mid_rows if r.pass_answer)) / len(mid_rows) if mid_rows else 0.0
        )
        acc_bot = (
            float(sum(1 for r in bot_rows if r.pass_answer)) / len(bot_rows) if bot_rows else 0.0
        )

        point_score = (acc_top + acc_bot) / 2.0 - acc_mid

        # Question-level scores for paired bootstrap CI
        q_map: dict[str, dict[str, list[bool]]] = {}
        for r in k_rows:
            qid = extract_question_id(r.item_id)
            if qid not in q_map:
                q_map[qid] = {"top": [], "middle": [], "bottom": []}
            if r.position in q_map[qid]:
                q_map[qid][r.position].append(r.pass_answer)

        q_scores: list[float] = []
        for _qid, pvals in q_map.items():
            q_top = float(sum(pvals["top"])) / len(pvals["top"]) if pvals["top"] else acc_top
            q_mid = (
                float(sum(pvals["middle"])) / len(pvals["middle"]) if pvals["middle"] else acc_mid
            )
            q_bot = (
                float(sum(pvals["bottom"])) / len(pvals["bottom"]) if pvals["bottom"] else acc_bot
            )
            q_scores.append((q_top + q_bot) / 2.0 - q_mid)

        if not q_scores:
            ci_lower, ci_upper = point_score, point_score
        elif all(s == q_scores[0] for s in q_scores):
            ci_lower, ci_upper = q_scores[0], q_scores[0]
        else:
            rng = np.random.default_rng(seed)
            arr = np.array(q_scores, dtype=float)
            n_q = len(arr)
            indices = rng.integers(0, n_q, size=(num_resamples, n_q))
            boot_means = np.mean(arr[indices], axis=1)
            alpha = (1.0 - confidence) / 2.0
            ci_lower = float(np.percentile(boot_means, alpha * 100.0))
            ci_upper = float(np.percentile(boot_means, (1.0 - alpha) * 100.0))

        results[k] = {
            "acc_top": acc_top,
            "acc_middle": acc_mid,
            "acc_bottom": acc_bot,
            "score": point_score,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "count_top": len(top_rows),
            "count_middle": len(mid_rows),
            "count_bottom": len(bot_rows),
        }

    return results
