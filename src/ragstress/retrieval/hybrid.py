"""Hybrid retrieval using Reciprocal Rank Fusion (SPEC §5.5, FR-RET-5)."""

from collections import defaultdict


def hybrid_rrf(
    ranked_id_lists: list[list[str]],
    k: int = 10,
    k_rrf: int = 60,
) -> list[tuple[str, float]]:
    """Compute Reciprocal Rank Fusion (RRF) over multiple ranked ID lists.

    Ties broken by ascending ID.
    Returns list of (id, rrf_score) sorted by score desc, id asc.
    """
    scores: dict[str, float] = defaultdict(float)

    for id_list in ranked_id_lists:
        for rank_1idx, item_id in enumerate(id_list, start=1):
            scores[item_id] += 1.0 / (k_rrf + rank_1idx)

    # Sort score desc, item_id asc
    sorted_items = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
    return sorted_items[:k]
