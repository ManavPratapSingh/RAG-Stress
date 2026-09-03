"""Vector store protocol and FlatStore implementation (SPEC §5.4, FR-RET-4)."""

from typing import Protocol

import numpy as np


class VectorStore(Protocol):
    def add(self, ids: list[str], vectors: np.ndarray) -> None: ...

    def search(self, query: np.ndarray, k: int) -> list[tuple[str, float]]:
        """Return (id, score) sorted by score desc, tie -> id asc."""
        ...


class FlatStore:
    """Exact cosine vector store backed by numpy dot products."""

    def __init__(self) -> None:
        self.ids: list[str] = []
        self.vectors: np.ndarray | None = None

    def add(self, ids: list[str], vectors: np.ndarray) -> None:
        if not ids:
            return
        if self.vectors is None:
            self.vectors = vectors.astype(np.float32)
            self.ids = list(ids)
        else:
            self.vectors = np.vstack([self.vectors, vectors.astype(np.float32)])
            self.ids.extend(ids)

    def search(self, query: np.ndarray, k: int) -> list[tuple[str, float]]:
        if self.vectors is None or len(self.ids) == 0:
            return []

        q = query.astype(np.float32)
        q_norm = np.linalg.norm(q)
        if q_norm > 0:
            q = q / q_norm

        # Cosine similarity via dot product (assumes added vectors are L2-normalized)
        scores = self.vectors @ q

        # Combine with ids for deterministic tie-breaking
        scored_items = [(self.ids[i], float(scores[i])) for i in range(len(self.ids))]
        scored_items.sort(key=lambda x: (-x[1], x[0]))

        return scored_items[:k]
