"""Exact cosine search over a small, in-memory NumPy index."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from rag_lab.chunking import Chunk


def normalise_vectors(vectors: ArrayLike) -> NDArray[np.float64]:
    array = np.asarray(vectors, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] == 0 or not np.isfinite(array).all():
        raise ValueError("Embeddings must be a finite matrix with a non-zero dimension.")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if not np.isfinite(norms).all() or np.any(norms == 0):
        raise ValueError("Embedding vectors must have finite, non-zero length.")
    return array / norms


@dataclass(frozen=True)
class SearchResult:
    chunk: Chunk
    score: float


class VectorIndex:
    def __init__(self, chunks: list[Chunk], embeddings: ArrayLike) -> None:
        self.chunks = tuple(chunks)
        self.embeddings = normalise_vectors(embeddings)
        if len(self.chunks) != len(self.embeddings):
            raise ValueError("Each chunk must have exactly one embedding.")
        self.embeddings.flags.writeable = False

    def search(self, query: ArrayLike, top_k: int = 5) -> list[SearchResult]:
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("Top-k must be a positive integer.")
        vector = np.asarray(query, dtype=np.float64)
        if vector.ndim != 1 or vector.shape[0] != self.embeddings.shape[1]:
            raise ValueError("Query must be a vector matching the index dimension.")
        vector = normalise_vectors(vector[None, :])[0]
        scores = np.clip(self.embeddings @ vector, -1.0, 1.0)
        # Stable sort makes tied scores retain ingestion order.
        order = np.argsort(-scores, kind="stable")[:top_k]
        return [SearchResult(self.chunks[i], float(scores[i])) for i in order]
