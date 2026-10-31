"""Cosine semantics without a model or network connection."""

import numpy as np
import pytest

from rag_lab.chunking import Chunk
from rag_lab.retrieval import VectorIndex


def chunks(count):
    return [Chunk(f"id-{i}", f"Passage {i}", "source.pdf", 2, 0, 9) for i in range(count)]


def test_ranking_scores_and_metadata():
    passages = chunks(4)
    index = VectorIndex(passages, [[0, 7], [3, 0], [-1, 0], [1, 1]])
    results = index.search([9, 0])
    assert [result.chunk for result in results] == [passages[i] for i in [1, 3, 0, 2]]
    assert [result.score for result in results] == pytest.approx([1, 2**-0.5, 0, -1])
    assert results[0].chunk.page == 2
    assert results[0].chunk.filename == "source.pdf"


def test_top_k_and_stable_ties():
    index = VectorIndex(chunks(3), [[1, 0]] * 3)
    assert [result.chunk.identifier for result in index.search([1, 0], 2)] == ["id-0", "id-1"]
    assert len(index.search([1, 0], 20)) == 3


def test_empty_index():
    index = VectorIndex([], np.empty((0, 2)))
    assert index.search([1, 0]) == []


@pytest.mark.parametrize(
    "vectors", [[], [[0, 0]], [[np.nan, 1]], [[np.inf, 1]], [[]], [1, 0], [[[1, 0]]]]
)
def test_invalid_index_vectors(vectors):
    with pytest.raises(ValueError):
        VectorIndex(chunks(1), vectors)


@pytest.mark.parametrize("query", [[], [0, 0], [np.nan, 0], [np.inf, 0], [1], [[1, 0]]])
def test_invalid_query_vectors(query):
    index = VectorIndex(chunks(1), [[1, 0]])
    with pytest.raises(ValueError):
        index.search(query)


@pytest.mark.parametrize("top_k", [0, -1, 1.5, True])
def test_invalid_top_k(top_k):
    with pytest.raises(ValueError, match="Top-k"):
        VectorIndex(chunks(1), [[1, 0]]).search([1, 0], top_k)


def test_chunk_vector_count_must_match():
    with pytest.raises(ValueError, match="Each chunk"):
        VectorIndex(chunks(2), [[1, 0]])


def test_index_owns_vectors():
    vectors = np.array([[1.0, 0.0]])
    index = VectorIndex(chunks(1), vectors)
    vectors[0] = [0, 1]
    assert index.search([1, 0])[0].score == 1
