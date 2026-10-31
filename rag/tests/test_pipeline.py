"""Exercise indexing and replacement with a deterministic fake encoder."""

import numpy as np
import pytest

from rag_lab.pipeline import RetrievalPipeline


@pytest.fixture
def files(tmp_path):
    paths = [tmp_path / "cat.md", tmp_path / "dog.txt"]
    for path, text in zip(paths, ["cat naps", "dog runs"], strict=True):
        path.write_text(text)
    return paths


def test_index_query_and_replacement(files):
    calls = []

    def encode(texts):
        calls.append(texts)
        return np.array([[1, 0] if "cat" in text else [0, 1] for text in texts])

    pipeline = RetrievalPipeline(encode)
    assert pipeline.index_documents(files) == 2
    assert pipeline.document_count == 2
    assert pipeline.search("cat", 1)[0].chunk.filename == "cat.md"
    assert calls == [["cat naps", "dog runs"], ["cat"]]
    pipeline.index_documents(files[1:])
    assert pipeline.document_count == 1
    assert len(pipeline.index.chunks) == 1
    assert pipeline.search("cat")[0].chunk.filename == "dog.txt"


def test_failed_replacement_preserves_collection(files):
    pipeline = RetrievalPipeline(lambda texts: [[1, 0] for _ in texts])
    pipeline.index_documents(files)
    old_index = pipeline.index
    with pytest.raises(ValueError):
        pipeline.index_documents([files[0].with_suffix(".invalid")])
    assert pipeline.index is old_index
    assert pipeline.document_count == 2


@pytest.mark.parametrize("query", ["", " \t", None])
def test_empty_query(files, query):
    def no_embedding(_):
        pytest.fail("Invalid input must not trigger embedding")

    with pytest.raises(ValueError, match="non-empty"):
        RetrievalPipeline(no_embedding).search(query)


def test_search_before_indexing():
    with pytest.raises(ValueError, match="Index documents"):
        RetrievalPipeline().search("query")


def test_index_no_documents():
    with pytest.raises(ValueError, match="Upload"):
        RetrievalPipeline().index_documents([])


def test_duplicate_names_have_distinct_ids(files):
    pipeline = RetrievalPipeline(lambda texts: [[1, 0] for _ in texts])
    pipeline.index_documents([files[0], files[0]])
    assert len({chunk.identifier for chunk in pipeline.index.chunks}) == 2


def test_default_pipeline_attaches_loaded_model_diagnostics(files, monkeypatch):
    from rag_lab import embeddings

    class FakeModel:
        max_seq_length = 3

        def encode(self, texts, **kwargs):
            return [[1, 0] for _ in texts]

        def tokenizer(self, texts, **kwargs):
            assert kwargs["add_special_tokens"] is True
            assert kwargs["truncation"] is False
            return {"input_ids": [[101, *text.split(), 102] for text in texts]}

    monkeypatch.setattr(embeddings, "_load_model", FakeModel)
    pipeline = RetrievalPipeline()
    pipeline.index_documents(files)
    assert pipeline.insights.model == embeddings.DEFAULT_MODEL
    assert pipeline.insights.token_limit == 3
    assert pipeline.insights.truncated == 2
    result = pipeline.search("cat", 1)[0]
    assert result.chunk.diagnostics.original_tokens == 4
    assert result.chunk.diagnostics.processed_tokens == 3
    old_index, old_insights = pipeline.index, pipeline.insights
    pipeline.diagnoser = lambda texts: []
    with pytest.raises(ValueError):
        pipeline.index_documents(files)
    assert pipeline.index is old_index
    assert pipeline.insights is old_insights
