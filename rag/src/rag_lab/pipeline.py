"""Gradio-independent ingestion and search orchestration."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from time import perf_counter

import numpy as np
from numpy.typing import ArrayLike, NDArray

from rag_lab.chunking import TokenDiagnostics, chunk_document, validate_parameters
from rag_lab.context import AssembledContext, assemble_context
from rag_lab.documents import load_document
from rag_lab.embeddings import DEFAULT_MODEL, diagnose_texts, encode_texts
from rag_lab.generation import ABSTENTION, LocalGenerator, check_citations
from rag_lab.insights import IndexInsights
from rag_lab.retrieval import SearchResult, VectorIndex


class IngestionPipeline:
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


class RetrievalPipeline(IngestionPipeline):
    """Retain the existing index/search API while exposing query stages."""

    def embed_query(self, query: str) -> NDArray[np.float64]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Enter a non-empty search query.")
        if self.index is None:
            raise ValueError("Index documents before searching.")
        return np.asarray(self.encoder([query]), dtype=np.float64)[0]

    def retrieve(self, vector: ArrayLike, top_k: int = 5) -> list[SearchResult]:
        if self.index is None:
            raise ValueError("Index documents before searching.")
        return self.index.search(vector, top_k)

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        return self.retrieve(self.embed_query(query), top_k)


@dataclass(frozen=True)
class QueryDiagnostics:
    durations: dict[str, float | None]
    retrieved: tuple[tuple[str, float], ...]
    model: str | None
    prompt_tokens: int | None
    context_window: int | None
    max_answer_tokens: int | None


@dataclass(frozen=True)
class QueryResult:
    results: list[SearchResult]
    context: AssembledContext
    answer: str | None
    status: str
    message: str
    citations: tuple[str, ...]
    warnings: tuple[str, ...]
    diagnostics: QueryDiagnostics


class QueryPipeline:
    """Explicit query stages; generation failure never discards retrieved evidence."""

    def __init__(
        self,
        retrieval: RetrievalPipeline,
        generator: LocalGenerator | None = None,
        clock: Callable[[], float] = perf_counter,
    ) -> None:
        self.retrieval = retrieval
        self.generator = generator
        self.clock = clock

    def run(self, question: str, top_k: int = 5) -> QueryResult:
        started = self.clock()
        durations: dict[str, float | None] = dict.fromkeys(
            ("query_embedding", "retrieval", "context_assembly", "model_loading", "generation")
        )

        def stage[T](name: str, action: Callable[[], T]) -> T:
            before = self.clock()
            try:
                return action()
            finally:
                durations[name] = (durations[name] or 0.0) + self.clock() - before

        vector = stage("query_embedding", lambda: self.retrieval.embed_query(question))
        results = stage("retrieval", lambda: self.retrieval.retrieve(vector, top_k))
        context = stage("context_assembly", lambda: assemble_context(results))
        answer = None
        citations: tuple[str, ...] = ()
        warnings: tuple[str, ...] = ()
        status, message = "retrieval_only", "Answer generation skipped."
        prepared = None
        generator = self.generator
        if generator is not None:
            if not context.sources:
                status, message, answer = "insufficient_context", ABSTENTION, ABSTENTION
            else:
                try:
                    stage("model_loading", generator.load)
                    prepared = stage(
                        "context_assembly", lambda: generator.prepare(question, results)
                    )
                    context = prepared.context
                    if not context.sources:
                        status, message, answer = "insufficient_context", ABSTENTION, ABSTENTION
                    else:
                        answer = stage("generation", lambda: generator.generate(prepared))
                        if not answer or not answer.strip():
                            raise ValueError("The local model returned an empty answer.")
                        citations, warnings = check_citations(answer, context)
                        status = "insufficient_context" if answer == ABSTENTION else "generated"
                        message = (
                            "Model reported insufficient evidence."
                            if answer == ABSTENTION
                            else ("Answer generated. Check claims against the cited passages.")
                        )
                except (ValueError, OSError, RuntimeError, ImportError, TypeError, KeyError) as exc:
                    answer = None
                    status, message = (
                        "generation_unavailable",
                        f"Answer generation unavailable: {exc}",
                    )
        durations["total"] = self.clock() - started
        return QueryResult(
            results,
            context,
            answer,
            status,
            message,
            citations,
            warnings,
            QueryDiagnostics(
                durations,
                tuple((result.chunk.identifier, result.score) for result in results),
                generator.identifier if generator is not None else None,
                len(prepared.tokens) if prepared else None,
                prepared.context_window if prepared else None,
                prepared.max_answer_tokens if prepared else None,
            ),
        )
