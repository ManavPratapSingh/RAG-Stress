"""Retrieval evaluation metrics (SPEC §5.6, FR-RET-6)."""

import math

from pydantic import BaseModel


class RetrievalMetrics(BaseModel):
    hit: float
    recall: float
    precision: float
    mrr: float
    ndcg: float
    distractor_rate: float


def dcg_at_k(binary_relevance: list[int], k: int) -> float:
    """Discounted Cumulative Gain with log2(rank + 1) discount."""
    dcg = 0.0
    for i, rel in enumerate(binary_relevance[:k], start=1):
        if rel > 0:
            dcg += rel / math.log2(i + 1)
    return dcg


def ndcg_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    """Normalized Discounted Cumulative Gain with binary relevance."""
    if not relevant:
        return 0.0
    binary_rel = [1 if item in relevant else 0 for item in ranked[:k]]
    actual_dcg = dcg_at_k(binary_rel, k)

    # Ideal ranking: all relevant items at the top
    ideal_count = min(len(relevant), k)
    ideal_rel = [1] * ideal_count
    ideal_dcg = dcg_at_k(ideal_rel, k)

    if ideal_dcg == 0.0:
        return 0.0
    return actual_dcg / ideal_dcg


def compute_retrieval_metrics(
    ranked: list[str],
    relevant: set[str],
    k: int,
    noise_pool: set[str] | None = None,
) -> RetrievalMetrics:
    """Compute Hit@K, Recall@K, Precision@K, MRR, nDCG@K, and distractor_rate@K.

    For unanswerable queries (relevant is empty), Hit, Recall, Precision, MRR,
    and nDCG are set to 0.0, and distractor_rate is reported.
    """
    top_k = ranked[:k]
    noise = noise_pool or set()
    distractors_in_top_k = sum(1 for item in top_k if item in noise)
    distractor_rate = (distractors_in_top_k / k) if k > 0 else 0.0

    if not relevant:
        return RetrievalMetrics(
            hit=0.0,
            recall=0.0,
            precision=0.0,
            mrr=0.0,
            ndcg=0.0,
            distractor_rate=distractor_rate,
        )

    matched = [item for item in top_k if item in relevant]
    hit = 1.0 if len(matched) > 0 else 0.0
    recall = len(matched) / len(relevant)
    precision = len(matched) / k

    # MRR: reciprocal rank of first relevant item in top-K (1-indexed)
    mrr = 0.0
    for rank, item in enumerate(top_k, start=1):
        if item in relevant:
            mrr = 1.0 / rank
            break

    ndcg = ndcg_at_k(top_k, relevant, k)

    return RetrievalMetrics(
        hit=hit,
        recall=recall,
        precision=precision,
        mrr=mrr,
        ndcg=ndcg,
        distractor_rate=distractor_rate,
    )
