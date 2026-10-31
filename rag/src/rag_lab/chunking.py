"""Deterministic character and sentence-aware windows with exact source offsets."""

import re
from bisect import bisect_right
from dataclasses import dataclass

from rag_lab.documents import Document

STRATEGIES = ("Character-based", "Sentence-aware")


@dataclass(frozen=True)
class TokenDiagnostics:
    original_tokens: int
    processed_tokens: int
    token_limit: int

    @property
    def truncated(self) -> bool:
        return self.original_tokens > self.token_limit


@dataclass(frozen=True)
class Chunk:
    identifier: str
    text: str
    filename: str
    page: int | None
    start: int
    end: int
    diagnostics: TokenDiagnostics | None = None


def validate_parameters(chunk_size: int, overlap: int) -> None:
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("Chunk size must be a positive integer (characters).")
    if isinstance(overlap, bool) or not isinstance(overlap, int) or not 0 <= overlap < chunk_size:
        raise ValueError("Overlap must be an integer from zero to less than the chunk size.")


def chunk_document(
    document: Document,
    chunk_size: int = 800,
    overlap: int = 120,
    document_id: int = 1,
    strategy: str = "Character-based",
) -> list[Chunk]:
    """Prefer whitespace in the latter half of a window, without losing characters.

    Offsets are Python character offsets within a document part (a PDF page).
    Overlap never crosses page boundaries. Whitespace is retained verbatim.
    Sentence-aware windows prefer punctuation, paragraph and list boundaries.
    They repeat complete trailing units within the overlap budget (possibly zero).
    Oversized units fall back to whitespace or hard character windows.
    """
    validate_parameters(chunk_size, overlap)
    if strategy not in STRATEGIES:
        raise ValueError("Choose Character-based or Sentence-aware chunking.")
    chunks = []
    for part in document.parts:
        if not part.text.strip():
            continue
        # Blank lines and list/heading starts are boundaries; ordinary PDF line wraps
        # stay inside a sentence. This is a heuristic, not linguistic parsing.
        boundaries = (
            sorted(
                {0, len(part.text)}
                | {
                    match.end()
                    for match in re.finditer(
                        r"[.!?][\"”’\x27)\]]*\s+|\n[ \t]*\n+|"
                        r"\n(?=[ \t]*(?:[-*+] |\d+[.)] |#{1,6} ))",
                        part.text,
                    )
                }
            )
            if strategy == "Sentence-aware"
            else []
        )
        start = 0
        while start < len(part.text):
            end = min(start + chunk_size, len(part.text))
            if end < len(part.text) and boundaries:
                candidate = boundaries[bisect_right(boundaries, end) - 1]
                if candidate > start:
                    end = candidate
            if end < len(part.text) and (not boundaries or end == start + chunk_size):
                minimum = start + max(chunk_size // 2, overlap + 1)
                for boundary in range(end, minimum - 1, -1):
                    if part.text[boundary - 1].isspace():
                        end = boundary
                        break
            chunks.append(
                Chunk(
                    f"doc-{document_id:04d}-chunk-{len(chunks) + 1:04d}",
                    part.text[start:end],
                    document.filename,
                    part.page,
                    start,
                    end,
                )
            )
            if end == len(part.text):
                break
            next_start = end - overlap
            if boundaries:
                # Include only complete trailing units within the overlap budget.
                boundary_index = bisect_right(boundaries, max(start, next_start - 1))
                next_start = min(boundaries[boundary_index], end)
                following_end = boundaries[bisect_right(boundaries, end)]
                if following_end - end <= chunk_size < following_end - next_start:
                    # Drop overlap when it would prevent the next whole unit fitting.
                    next_start = end
            start = next_start
    return chunks
