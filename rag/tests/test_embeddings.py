"""Embedding adapter options with a fake model, never real inference."""

import numpy as np
import pytest

from rag_lab import embeddings


def test_batch_encoding_requests_normalised_numpy_vectors(monkeypatch):
    calls = []

    class FakeModel:
        def encode(self, texts, **kwargs):
            calls.append((texts, kwargs))
            return [[1, 0] for _ in texts]

    monkeypatch.setattr(embeddings, "_load_model", FakeModel)
    # Whitespace-only windows inside otherwise valid documents retain source text.
    result = embeddings.encode_texts(["one", "  "])
    assert result.dtype == np.float32
    assert result.shape == (2, 2)
    assert calls[0][0] == ["one", "  "]
    assert calls[0][1]["normalize_embeddings"] is True
    assert calls[0][1]["convert_to_numpy"] is True
    assert calls[0][1]["batch_size"] == 32


def test_load_error_is_actionable(monkeypatch):
    def fail():
        raise OSError("offline")

    monkeypatch.setattr(embeddings, "_load_model", fail)
    with pytest.raises(RuntimeError, match="first use, internet access"):
        embeddings.encode_texts(["text"])


def test_token_diagnostics_use_loaded_tokenizer_and_special_tokens(monkeypatch):
    calls = []

    class FakeModel:
        max_seq_length = 5

        def tokenizer(self, texts, **kwargs):
            calls.append((texts, kwargs))
            return {"input_ids": [[101] + list(range(len(text.split()))) + [102] for text in texts]}

    monkeypatch.setattr(embeddings, "_load_model", FakeModel)
    diagnostics = embeddings.diagnose_texts(["one two", "one two three", "one two three four", ""])
    assert [d.original_tokens for d in diagnostics] == [4, 5, 6, 2]
    assert [d.processed_tokens for d in diagnostics] == [4, 5, 5, 2]
    assert [d.truncated for d in diagnostics] == [False, False, True, False]
    assert all(d.token_limit == 5 for d in diagnostics)
    assert calls[0][1]["add_special_tokens"] is True
    assert calls[0][1]["truncation"] is False
    assert calls[0][1]["padding"] is False
    assert embeddings.diagnose_texts([]) == []
    assert len(calls) == 1
