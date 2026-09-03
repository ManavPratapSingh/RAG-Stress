"""Custom BM25 retriever implementation (SPEC §5.2, FR-RET-2)."""

import math
import re
from collections import Counter

from pydantic import BaseModel

from ragstress.retrieval.chunker import Chunk


class RetrievedChunk(BaseModel):
    doc_id: str
    chunk_id: str
    score: float
    rank: int


def tokenize(text: str) -> list[str]:
    """Lowercase word tokenizer matching `\\w+`."""
    return re.findall(r"\w+", text.lower())


class BM25Index:
    """In-memory BM25 index with stable tie-breaking."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.chunks: list[Chunk] = []
        self.doc_len: list[int] = []
        self.avgdl: float = 0.0
        self.doc_freqs: Counter[str] = Counter()
        self.term_freqs: list[Counter[str]] = []
        self.N: int = 0

    def index(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        self.N = len(chunks)
        self.doc_len = []
        self.term_freqs = []
        self.doc_freqs = Counter()

        total_words = 0
        for chunk in chunks:
            tokens = tokenize(chunk.text)
            length = len(tokens)
            self.doc_len.append(length)
            total_words += length
            tf = Counter(tokens)
            self.term_freqs.append(tf)
            for token in tf:
                self.doc_freqs[token] += 1

        self.avgdl = (total_words / self.N) if self.N > 0 else 0.0

    def _idf(self, term: str) -> float:
        n = self.doc_freqs.get(term, 0)
        # SPEC §5.2 formula: ln(1 + (N - n + 0.5) / (n + 0.5))
        return math.log(1 + (self.N - n + 0.5) / (n + 0.5))

    def search(self, query: str, k: int = 10) -> list[RetrievedChunk]:
        query_tokens = tokenize(query)
        scores: list[tuple[float, str, Chunk]] = []

        for idx, chunk in enumerate(self.chunks):
            dl = self.doc_len[idx]
            len_norm = 1 - self.b + self.b * (dl / self.avgdl) if self.avgdl > 0 else 1.0
            score = 0.0
            tf = self.term_freqs[idx]

            for term in query_tokens:
                freq = tf.get(term, 0)
                if freq > 0:
                    idf = self._idf(term)
                    term_score = idf * (freq * (self.k1 + 1)) / (freq + self.k1 * len_norm)
                    score += term_score

            scores.append((score, chunk.chunk_id, chunk))

        # Tie-breaking: score desc, chunk_id asc
        scores.sort(key=lambda item: (-item[0], item[1]))

        results: list[RetrievedChunk] = []
        for rank, (score, _, chunk) in enumerate(scores[:k], start=1):
            results.append(
                RetrievedChunk(
                    doc_id=chunk.doc_id,
                    chunk_id=chunk.chunk_id,
                    score=float(score),
                    rank=rank,
                )
            )
        return results
