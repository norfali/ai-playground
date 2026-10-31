"""Never allow an automated test to load or download real model weights."""

import pytest

from rag_lab import embeddings


@pytest.fixture(autouse=True)
def no_model_downloads(monkeypatch):
    def fail():
        pytest.fail("Tests must replace the embedding model with a fake")

    monkeypatch.setattr(embeddings, "_load_model", fail)
