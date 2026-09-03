"""Embedder protocol and deterministic HashEmbedder (SPEC §5.3, FR-RET-3)."""

import hashlib
from typing import Protocol

import numpy as np


class Embedder(Protocol):
    name: str
    dim: int

    def embed(self, texts: list[str]) -> np.ndarray:
        """Return float32, L2-normalized array of shape (n, dim)."""
        ...


class HashEmbedder:
    """Deterministic, dependency-free feature-hashing embedder.

    Hashes word unigrams, word bigrams, and character 3-grams into `dim` buckets
    with signed hashing, followed by L2-normalization.
    """

    def __init__(self, dim: int = 512, name: str = "hash-512"):
        self.dim = dim
        self.name = name

    def _hash_feature(self, feature: str) -> tuple[int, float]:
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        val = int.from_bytes(digest, "little", signed=True)
        bucket = abs(val) % self.dim
        sign = 1.0 if val >= 0 else -1.0
        return bucket, sign

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), self.dim), dtype=np.float32)

        for row, text in enumerate(texts):
            words = text.lower().split()

            # Word unigrams
            for w in words:
                b, s = self._hash_feature(f"u:{w}")
                vectors[row, b] += s

            # Word bigrams
            for i in range(len(words) - 1):
                bg = f"b:{words[i]}_{words[i + 1]}"
                b, s = self._hash_feature(bg)
                vectors[row, b] += s

            # Character 3-grams
            for i in range(len(text) - 2):
                c3 = f"c:{text[i : i + 3].lower()}"
                b, s = self._hash_feature(c3)
                vectors[row, b] += s

        # L2-normalization
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        return (vectors / norms).astype(np.float32)
