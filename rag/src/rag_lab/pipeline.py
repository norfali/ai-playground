"""Gradio-independent ingestion and search orchestration."""

from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path

from numpy.typing import ArrayLike

from rag_lab.chunking import TokenDiagnostics, chunk_document, validate_parameters
from rag_lab.documents import load_document
from rag_lab.embeddings import DEFAULT_MODEL, diagnose_texts, encode_texts
from rag_lab.insights import IndexInsights
from rag_lab.retrieval import SearchResult, VectorIndex


class RetrievalPipeline:
    def __init__(
        self,
        encoder: Callable[[list[str]], ArrayLike] = encode_texts,
        diagnoser: Callable[[list[str]], list[TokenDiagnostics]] | None = None,
    ) -> None:
        self.encoder = encoder
        self.diagnoser = diagnoser or (diagnose_texts if encoder is encode_texts else None)
        self.insights: IndexInsights | None = None
        self.index: VectorIndex | None = None
        self.document_count = 0

    def index_documents(
        self,
        paths: Sequence[str | Path],
        chunk_size: int = 800,
        overlap: int = 120,
        strategy: str = "Character-based",
    ) -> int:
        """Build a replacement collection atomically; failures leave the prior index intact."""
        validate_parameters(chunk_size, overlap)
        if not paths:
            raise ValueError("Upload at least one document before indexing.")
        documents = [load_document(path) for path in paths]
        chunks = [
            chunk
            for number, document in enumerate(documents, start=1)
            for chunk in chunk_document(document, chunk_size, overlap, number, strategy)
        ]
        texts = [chunk.text for chunk in chunks]
        vectors = self.encoder(texts)
        if self.diagnoser is not None:
            diagnostics = self.diagnoser(texts)
            chunks = [
                replace(chunk, diagnostics=diagnostic)
                for chunk, diagnostic in zip(chunks, diagnostics, strict=True)
            ]
        index = VectorIndex(chunks, vectors)
        insights = IndexInsights.from_chunks(
            chunks,
            len(documents),
            strategy,
            chunk_size,
            overlap,
            DEFAULT_MODEL if self.encoder is encode_texts else "Custom encoder",
        )
        self.index = index
        self.insights = insights
        self.document_count = len(documents)
        return len(chunks)

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Enter a non-empty search query.")
        if self.index is None:
            raise ValueError("Index documents before searching.")
        return self.index.search(self.encoder([query])[0], top_k)
