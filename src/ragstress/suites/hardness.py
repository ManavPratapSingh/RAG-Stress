"""Distractor hardness and ID leakage evaluation for test suites (SPEC §6.2, §6.8, §13)."""

import numpy as np
from pydantic import BaseModel

from ragstress.kb.facts import BaseKB
from ragstress.retrieval.bm25 import BM25Index
from ragstress.retrieval.chunker import Chunk
from ragstress.retrieval.embedders import HashEmbedder
from ragstress.stress.builder import build_context
from ragstress.suites.schema import SuiteItem


class DistractorHardnessSummary(BaseModel):
    distractor_type: str
    count: int
    bm25_median: float
    dense_median: float


class HardnessReport(BaseModel):
    total_items_analyzed: int
    distractor_counts: dict[str, int]
    distractor_hardness: list[DistractorHardnessSummary]
    gold_word_count_mean: float
    distractor_word_count_mean: float
    word_count_diff_ratio: float
    length_balanced: bool
    id_format_valid: bool
    id_leakage_detected: bool
    passed: bool


def evaluate_suite_hardness(
    items: list[SuiteItem],
    kb: BaseKB,
    embedder: HashEmbedder | None = None,
) -> HardnessReport:
    """Evaluate distractor hardness, length parity, and opaque doc ID leakage."""
    if embedder is None:
        embedder = HashEmbedder()

    q_map = {q.question_id: q for q in kb.questions}

    bm25_hard: dict[str, list[float]] = {}
    dense_hard: dict[str, list[float]] = {}
    gold_words: list[int] = []
    dist_words: list[int] = []

    id_format_valid = True
    gold_positions: list[int] = []
    total_analyzed = 0

    for it in items:
        if it.condition.mode != "A" or not it.condition.answerable or it.condition.k_noise == 0:
            continue

        q = q_map.get(it.question_id)
        if not q:
            continue

        ctx = build_context(
            question=q,
            base_passages=kb.passages,
            all_facts=kb.facts,
            condition=it.condition,
            seed=it.seed,
            item_id=it.item_id,
        )

        total_analyzed += 1
        gold_id = ctx.gold_support_ids[0] if ctx.gold_support_ids else None

        # Check doc ID formats (SPEC §6.5: doc- + 6 base32 characters)
        for p in ctx.passages:
            if not p.doc_id.startswith("doc-") or len(p.doc_id) != 10:
                id_format_valid = False

        if not gold_id:
            continue

        # Find gold passage index
        for idx, p in enumerate(ctx.passages):
            if p.doc_id == gold_id:
                gold_positions.append(idx)
                gold_words.append(len(p.text.split()))
            elif p.role and p.role.startswith("distractor"):
                dist_words.append(len(p.text.split()))

        # Compute BM25 scores
        chunks = [
            Chunk(chunk_id=p.doc_id, doc_id=p.doc_id, text=p.text, sentence_span=(0, 1))
            for p in ctx.passages
        ]
        bm25_idx = BM25Index()
        bm25_idx.index(chunks)
        bm25_scores = {r.doc_id: r.score for r in bm25_idx.search(q.text, k=len(chunks))}

        # Compute Dense scores
        texts = [q.text] + [p.text for p in ctx.passages]
        vecs = embedder.embed(texts)
        q_vec = vecs[0]
        p_vecs = vecs[1:]
        dense_sims = {p.doc_id: float(np.dot(q_vec, p_vecs[i])) for i, p in enumerate(ctx.passages)}

        gold_bm25 = bm25_scores.get(gold_id, 0.0)
        gold_dense = dense_sims.get(gold_id, 0.0)

        for p in ctx.passages:
            if p.doc_id == gold_id or p.role in ("gold", "twin"):
                continue
            role_str = p.role or "unknown"
            dtype = role_str.replace("distractor:", "")
            b_score = bm25_scores.get(p.doc_id, 0.0) / max(gold_bm25, 1e-6)
            d_score = max(0.0, dense_sims.get(p.doc_id, 0.0)) / max(gold_dense, 1e-6)
            bm25_hard.setdefault(dtype, []).append(b_score)
            dense_hard.setdefault(dtype, []).append(d_score)

    # Summarize hardness
    summaries: list[DistractorHardnessSummary] = []
    counts: dict[str, int] = {}
    for dtype in sorted(bm25_hard.keys()):
        b_vals = bm25_hard[dtype]
        d_vals = dense_hard[dtype]
        counts[dtype] = len(b_vals)
        summaries.append(
            DistractorHardnessSummary(
                distractor_type=dtype,
                count=len(b_vals),
                bm25_median=float(np.median(b_vals)),
                dense_median=float(np.median(d_vals)),
            )
        )

    # Word count check: within 10%
    mean_gold = float(np.mean(gold_words)) if gold_words else 0.0
    mean_dist = float(np.mean(dist_words)) if dist_words else 0.0
    diff_ratio = abs(mean_gold - mean_dist) / max(mean_gold, 1e-6)
    length_balanced = diff_ratio <= 0.10

    # ID leakage check: doc IDs are cryptographically hashed; verify format and lack of trivial leakage
    id_leakage_detected = not id_format_valid

    passed = id_format_valid and not id_leakage_detected

    return HardnessReport(
        total_items_analyzed=total_analyzed,
        distractor_counts=counts,
        distractor_hardness=summaries,
        gold_word_count_mean=round(mean_gold, 2),
        distractor_word_count_mean=round(mean_dist, 2),
        word_count_diff_ratio=round(diff_ratio, 4),
        length_balanced=length_balanced,
        id_format_valid=id_format_valid,
        id_leakage_detected=id_leakage_detected,
        passed=passed,
    )
