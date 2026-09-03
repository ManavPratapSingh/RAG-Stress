import math

import numpy as np

from ragstress.retrieval.bm25 import BM25Index
from ragstress.retrieval.chunker import Chunk, chunk_passage
from ragstress.retrieval.embedders import HashEmbedder
from ragstress.retrieval.hybrid import hybrid_rrf
from ragstress.retrieval.metrics import (
    compute_retrieval_metrics,
)
from ragstress.retrieval.stores import FlatStore


def test_chunker_sentence_boundaries() -> None:
    text = (
        "First sentence here. Second sentence with words. "
        "Third sentence is here. Fourth sentence concludes."
    )
    # With max_words = 6, it should chunk without splitting any sentence
    chunks = chunk_passage("doc-001", "Doc Title", text, max_words=6)
    assert len(chunks) > 1
    recombined = " ".join(c.text for c in chunks)
    assert "First sentence here." in recombined
    assert "First sentence here." in chunks[0].text
    for c in chunks:
        assert c.chunk_id.startswith("doc-001#c")
        assert c.doc_id == "doc-001"


def test_bm25_hand_computed() -> None:
    """Requirement FR-RET-2: test against hand-computed 3-doc example."""
    # Docs:
    # d0: "cat dog" (len 2)
    # d1: "cat mouse" (len 2)
    # d2: "dog dog dog" (len 3)
    # N = 3, avgdl = (2 + 2 + 3) / 3 = 7/3 = 2.3333333333333335
    # Query: "cat dog"
    # df("cat") = 2, df("dog") = 2
    # IDF formula: ln(1 + (N - n + 0.5) / (n + 0.5))
    # For n=2, N=3: ln(1 + (3 - 2 + 0.5) / (2 + 0.5)) = ln(1 + 1.5 / 2.5) = ln(1.6) ≈ 0.470003629
    # k1 = 1.5, b = 0.75
    # For d0:
    # |d0| = 2
    # len_norm = 1 - b + b * (|d0| / avgdl) = 1 - 0.75 + 0.75 * (2 / (7/3)) = 0.25 + 0.75 * 6/7 = 0.25 + 0.642857 = 0.892857
    # tf(cat) = 1 -> tf_norm = 1 * 2.5 / (1 + 1.5 * 0.892857) = 2.5 / (1 + 1.3392857) = 2.5 / 2.3392857 ≈ 1.068699
    # tf(dog) = 1 -> same = 1.068699
    # score(d0) = ln(1.6) * 1.068699 * 2 ≈ 0.470003629 * 2.137398 ≈ 1.004584

    chunks = [
        Chunk(chunk_id="c0", doc_id="d0", text="cat dog", sentence_span=(0, 1)),
        Chunk(chunk_id="c1", doc_id="d1", text="cat mouse", sentence_span=(0, 1)),
        Chunk(chunk_id="c2", doc_id="d2", text="dog dog dog", sentence_span=(0, 1)),
    ]
    bm25 = BM25Index()
    bm25.index(chunks)

    results = bm25.search("cat dog", k=3)
    assert len(results) == 3
    # Top should be c0 (has both terms)
    assert results[0].chunk_id == "c0"

    expected_idf = math.log(1 + (3 - 2 + 0.5) / (2 + 0.5))
    avgdl = 7.0 / 3.0
    len_norm_0 = 1 - 0.75 + 0.75 * (2.0 / avgdl)
    tf_norm_0 = (1.0 * 2.5) / (1.0 + 1.5 * len_norm_0)
    expected_score_0 = expected_idf * tf_norm_0 * 2

    assert math.isclose(results[0].score, expected_score_0, rel_tol=1e-5)


def test_bm25_stable_tie_breaks() -> None:
    # Two identical docs with different IDs should break ties by ascending chunk_id
    chunks = [
        Chunk(chunk_id="c_beta", doc_id="d2", text="alpha", sentence_span=(0, 1)),
        Chunk(chunk_id="c_alpha", doc_id="d1", text="alpha", sentence_span=(0, 1)),
    ]
    bm25 = BM25Index()
    bm25.index(chunks)
    res = bm25.search("alpha", k=2)
    assert res[0].chunk_id == "c_alpha"
    assert res[1].chunk_id == "c_beta"


def test_hash_embedder_and_flat_store() -> None:
    embedder = HashEmbedder(dim=512)
    texts = ["apple banana", "orange banana", "random entity policy"]
    vecs = embedder.embed(texts)
    assert vecs.shape == (3, 512)
    assert vecs.dtype == np.float32

    # Verify L2 normalized
    norms = np.linalg.norm(vecs, axis=1)
    for n in norms:
        assert math.isclose(float(n), 1.0, rel_tol=1e-5)

    store = FlatStore()
    store.add(["id0", "id1", "id2"], vecs)

    # Search query
    q_vec = embedder.embed(["apple banana"])
    results = store.search(q_vec[0], k=2)
    assert len(results) == 2
    assert results[0][0] == "id0"
    assert math.isclose(results[0][1], 1.0, rel_tol=1e-4)


def test_hybrid_rrf() -> None:
    # List 1: ["d0", "d1", "d2"]
    # List 2: ["d1", "d0", "d3"]
    # RRF score = sum(1 / (60 + rank))
    res = hybrid_rrf([["d0", "d1", "d2"], ["d1", "d0", "d3"]], k=3, k_rrf=60)
    # d0: 1/(60+1) + 1/(60+2) = 1/61 + 1/62 ≈ 0.016393 + 0.016129 = 0.032522
    # d1: 1/(60+2) + 1/(60+1) = identical score -> tie broken by id asc -> "d0" beats "d1"
    assert res[0][0] == "d0"
    assert res[1][0] == "d1"


def test_retrieval_metrics_worked_example() -> None:
    # Relevant docs: {"d_gold"}
    # Ranked: ["d_noise1", "d_gold", "d_noise2", "d_noise3", "d_noise4"]
    # At K=5:
    # Hit@5 = 1.0
    # Recall@5 = 1/1 = 1.0
    # Precision@5 = 1/5 = 0.2
    # MRR = 1/2 = 0.5
    # nDCG@5: rank of gold is 2 (1-indexed). DCG = 1 / log2(2 + 1) = 1 / log2(3) = 0.63092975
    # IDCG = 1 / log2(1 + 1) = 1 / log2(2) = 1.0 -> nDCG = 0.63092975
    ranked = ["d_noise1", "d_gold", "d_noise2", "d_noise3", "d_noise4"]
    metrics = compute_retrieval_metrics(
        ranked=ranked,
        relevant={"d_gold"},
        k=5,
        noise_pool={"d_noise1", "d_noise2", "d_noise3", "d_noise4"},
    )
    assert metrics.hit == 1.0
    assert metrics.recall == 1.0
    assert metrics.precision == 0.2
    assert metrics.mrr == 0.5
    assert math.isclose(metrics.ndcg, 1.0 / math.log2(3), rel_tol=1e-5)
    assert metrics.distractor_rate == 4.0 / 5.0
