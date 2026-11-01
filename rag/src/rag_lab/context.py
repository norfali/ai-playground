"""Deterministic context assembly with references to exact, unmodified chunks."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from rag_lab.retrieval import SearchResult


@dataclass(frozen=True)
class SourceReference:
    label: str
    result: SearchResult

    @property
    def header(self) -> str:
        chunk = self.result.chunk
        page = f", page {chunk.page}" if chunk.page is not None else ""
        return (
            f"[{self.label}] {chunk.filename}{page} | {chunk.identifier} | "
            f"characters {chunk.start}:{chunk.end}"
        )


@dataclass(frozen=True)
class AssembledContext:
    text: str
    sources: tuple[SourceReference, ...]
    omitted_ids: tuple[str, ...]


def assemble_context(
    results: Sequence[SearchResult],
    fits: Callable[[str], bool] | None = None,
) -> AssembledContext:
    """Include whole chunks in rank order, skipping those that do not fit.

    With no tokenizer budget (retrieval only), include all nonblank passages.
    Labels retain retrieval rank even when a passage is omitted. No source text
    is truncated or rewritten; offsets remain those of the original chunk.
    """
    sources: list[SourceReference] = []
    passages: list[str] = []
    omitted: list[str] = []
    for rank, result in enumerate(results, start=1):
        source = SourceReference(f"S{rank}", result)
        passage = f"{source.header}\n{result.chunk.text}"
        candidate = "\n\n".join([*passages, passage])
        if not result.chunk.text.strip() or (fits is not None and not fits(candidate)):
            omitted.append(result.chunk.identifier)
            continue
        passages.append(passage)
        sources.append(source)
    return AssembledContext("\n\n".join(passages), tuple(sources), tuple(omitted))
