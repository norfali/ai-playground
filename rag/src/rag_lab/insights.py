"""Deterministic measurements and explicitly qualified retrieval interpretations."""

from dataclasses import dataclass
from html import escape
from itertools import combinations

from rag_lab.chunking import Chunk
from rag_lab.retrieval import SearchResult


@dataclass(frozen=True)
class IndexInsights:
    documents: int
    chunks: int
    strategy: str
    chunk_size: int
    overlap: int
    model: str
    minimum: int
    maximum: int
    average: float
    token_limit: int | None
    diagnosed: int
    truncated: int

    @classmethod
    def from_chunks(
        cls,
        chunks: list[Chunk],
        documents: int,
        strategy: str,
        chunk_size: int,
        overlap: int,
        model: str,
    ) -> "IndexInsights":
        lengths = [len(chunk.text) for chunk in chunks]
        diagnostics = [chunk.diagnostics for chunk in chunks if chunk.diagnostics is not None]
        limits = {item.token_limit for item in diagnostics}
        return cls(
            documents,
            len(chunks),
            strategy,
            chunk_size,
            overlap,
            model,
            min(lengths, default=0),
            max(lengths, default=0),
            sum(lengths) / len(lengths) if lengths else 0,
            next(iter(limits)) if len(limits) == 1 else None,
            len(diagnostics),
            sum(item.truncated for item in diagnostics),
        )


def _sections(happened: str, meaning: list[str], experiment: str) -> str:
    """Render escaped prose; measurements can use a small, trusted HTML fragment."""
    return (
        "<h3>What happened?</h3>"
        + happened
        + "<h3>What does it mean?</h3>"
        + "".join(f"<p>{escape(paragraph)}</p>" for paragraph in meaning)
        + "<h3>What should I try next?</h3><p>"
        + escape(experiment)
        + "</p>"
    )


def render_summary_metrics(results: list[SearchResult], index: IndexInsights | None = None) -> str:
    """Keep search scores separate from index-wide tokenizer measurements."""
    top = f"{results[0].score:.4f}" if results else "Unavailable"
    margin = f"{results[0].score - results[1].score:.4f}" if len(results) > 1 else "Unavailable"
    truncation = (
        f"{index.truncated} / {index.diagnosed} diagnosed"
        if index is not None and index.diagnosed
        else "Unavailable"
    )
    return (
        '<dl class="insight-measurements" aria-label="Retrieval summary">'
        f"<div><dt>Top similarity</dt><dd>{top}</dd></div>"
        f"<div><dt>Score margin</dt><dd>{margin}</dd></div>"
        "<div><dt>Token truncation <small>Index-wide</small></dt>"
        f"<dd>{truncation}</dd></div></dl>"
    )


def suggest_experiment(
    index: IndexInsights | None,
    *,
    close_scores: bool = False,
    truncated: bool = False,
    repeated_source: bool = False,
) -> str:
    """Suggest one experiment using the indexed settings, within the UI ranges."""
    if index is None:
        return "Index configuration is unavailable; no configuration experiment can be suggested."
    if (
        index.truncated
        or truncated
        or (close_scores and not (repeated_source and index.overlap > 0))
    ):
        # Match the sliders' 50-character size and 10-character overlap steps.
        size = max(100, min(2000, (index.chunk_size * 5 // 8 // 50) * 50))
        if size < index.chunk_size:
            overlap = min(index.overlap, 500, ((size - 1) // 10) * 10)
            adjustment = (
                f" Also reduce overlap from {index.overlap} to {overlap} characters."
                if overlap != index.overlap
                else ""
            )
            observation = (
                "whether truncation decreases"
                if index.truncated or truncated
                else "whether the ranking changes"
            )
            return (
                f"Try reducing the chunk size from {index.chunk_size} to {size} characters."
                f"{adjustment} Re-index and repeat the same query to see {observation}."
            )
        if index.truncated or truncated:
            return (
                "Chunk size is already at the UI minimum (100 characters). "
                "No smaller-chunk experiment is available within these controls."
            )
    if repeated_source and index.overlap > 0:
        overlap = min(500, (index.overlap // 20) * 10)
        return (
            f"Try reducing overlap from {index.overlap} to {overlap} characters. "
            "Re-index and repeat the same query to compare source repetition."
        )
    if index.strategy == "Character-based":
        return (
            "Try sentence-aware chunking with the same size and overlap. "
            "Re-index and repeat the same query to compare how complete the passages are."
        )
    return "No further configuration experiment is indicated by these measurements."


def render_index_insights(index: IndexInsights) -> str:
    happened = (
        f"<p>{index.documents} documents · {index.chunks} chunks · {escape(index.strategy)}.</p>"
    )
    meaning = [
        "Character-based windows can split sentences."
        if index.strategy == "Character-based"
        else "Sentence-aware boundaries try to preserve sentences using a heuristic."
    ]
    if index.truncated:
        meaning.append(
            "Some chunks exceed the embedding token limit. Text beyond it may not "
            "contribute to embeddings; full passages remain visible."
        )
    if index.diagnosed < index.chunks:
        meaning.append("Token diagnostics are unavailable for some or all chunks.")
    return (
        '<section aria-label="Index measurements">'
        + render_summary_metrics([], index)
        + _sections(happened, meaning, suggest_experiment(index))
        + render_index_diagnostics(index)
        + "</section>"
    )


def render_index_diagnostics(index: IndexInsights) -> str:
    """Keep the current index configuration available without repeating explanations."""
    diagnostics = [
        f"{index.documents} documents · {index.chunks} chunks · {index.strategy}.",
        f"Configured size: {index.chunk_size} characters; overlap: {index.overlap} characters.",
        f"Chunk lengths: average {index.average:.1f}, minimum {index.minimum}, "
        f"maximum {index.maximum} characters.",
        f"Embedding model: {index.model}.",
    ]
    if index.token_limit is not None:
        diagnostics.append(
            f"Effective input limit: {index.token_limit} tokens (including special tokens)."
        )
    if index.diagnosed:
        diagnostics.append(
            f"{index.truncated} of {index.diagnosed} diagnosed chunks exceed their token limit."
        )
    if index.diagnosed < index.chunks:
        diagnostics.append("Token diagnostics are unavailable for some or all chunks.")
    return (
        '<details class="index-diagnostics"><summary>Indexing diagnostics</summary>'
        + "".join(f"<p>{escape(line)}</p>" for line in diagnostics)
        + "</details>"
    )


def overlap_fraction(first: Chunk, second: Chunk) -> float | None:
    """Source overlap / shorter passage length, only for the same document part.

    IDs generated by chunk_document encode document identity, unlike filenames.
    Unknown ID formats are not sufficient evidence for a source comparison.
    """
    first_id = first.identifier.partition("-chunk-")
    second_id = second.identifier.partition("-chunk-")
    if not first_id[1] or not second_id[1]:
        return None
    if first_id[0] != second_id[0] or first.page != second.page:
        return None
    length = min(first.end - first.start, second.end - second.start)
    if length <= 0:
        return None
    return max(0, min(first.end, second.end) - max(first.start, second.start)) / length


def render_search_insights(results: list[SearchResult], index: IndexInsights | None = None) -> str:
    if not results:
        return render_summary_metrics([], index) + _sections(
            "<p>No passages retrieved.</p>",
            ["There are no passages to compare."],
            "Index documents with readable text, then search again.",
        )
    margin = results[0].score - results[1].score if len(results) > 1 else None
    close_scores = margin is not None and margin <= 0.02
    meaning = []
    if close_scores:
        meaning.append(
            "Your top two passages have almost identical similarity scores. "
            "This does not establish equal relevance. The 0.02 cutoff is a display heuristic."
        )
    meaning.append(
        "Cosine similarity measures semantic alignment, not a calibrated confidence "
        "score or a relevance judgement."
    )
    diagnosed = [item.chunk.diagnostics for item in results if item.chunk.diagnostics is not None]
    truncated = sum(item.truncated for item in diagnosed)
    if truncated:
        meaning.append(
            f"{truncated} retrieved {'passage exceeds' if truncated == 1 else 'passages exceed'} "
            "the embedding input limit. "
            "Text beyond it may not contribute to their embeddings."
        )
    if len(diagnosed) < len(results):
        meaning.append("Token diagnostics are unavailable for some or all retrieved passages.")
    if index is not None and index.diagnosed < index.chunks:
        meaning.append(
            f"The index-wide truncation count covers {index.diagnosed} of {index.chunks} chunks; "
            "the remaining diagnostics are unavailable."
        )
    fractions = [
        fraction
        for a, b in combinations(results, 2)
        if (fraction := overlap_fraction(a.chunk, b.chunk)) is not None
    ]
    repeated_source = bool(fractions and max(fractions) >= 0.5)
    overlap = ""
    if fractions:
        overlap = (
            f"<p>Largest source overlap: {max(fractions):.1%} of the shorter passage "
            "among pairs from the same document and page.</p>"
        )
        if repeated_source:
            meaning.append("Repeated source text may reduce the variety of evidence.")
    happened = (
        f"<p>{len(results)} passages ranked by cosine similarity; "
        "ties retain ingestion order.</p>" + overlap
    )
    return (
        '<section aria-label="Search measurements">'
        + render_summary_metrics(results, index)
        + _sections(
            happened,
            meaning,
            suggest_experiment(
                index,
                close_scores=close_scores,
                truncated=bool(truncated),
                repeated_source=repeated_source,
            ),
        )
        + "</section>"
    )
