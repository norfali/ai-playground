"""Exact source reconstruction and deterministic chunk boundaries."""

import pytest

from rag_lab.chunking import chunk_document
from rag_lab.documents import Document, DocumentPart


def document(text):
    return Document("notes.md", (DocumentPart(text),))


def test_short_document():
    chunks = chunk_document(document("short text"))
    assert len(chunks) == 1
    assert chunks[0].text == "short text"
    assert chunks[0].start == 0
    assert chunks[0].end == 10


@pytest.mark.parametrize("size", [1, 7, 20, 81])
@pytest.mark.parametrize("overlap_fraction", [0, 0.25, 0.95])
@pytest.mark.parametrize("text", ["abcdefghijklmnop" * 40, "First word.\n\nOther words!  " * 30])
def test_no_text_lost_or_duplicated(size, overlap_fraction, text):
    overlap = int(size * overlap_fraction)
    chunks = chunk_document(document(text), size, overlap)
    assert chunks == chunk_document(document(text), size, overlap)
    assert all(0 < len(chunk.text) <= size for chunk in chunks)
    assert all(chunk.text == text[chunk.start : chunk.end] for chunk in chunks)
    assert chunks[0].start == 0
    assert chunks[-1].end == len(text)
    for previous, current in zip(chunks, chunks[1:], strict=False):
        assert current.start == previous.end - overlap
    reconstructed = chunks[0].text + "".join(chunk.text[overlap:] for chunk in chunks[1:])
    assert reconstructed == text


@pytest.mark.parametrize(
    "size,overlap",
    [(0, 0), (-1, 0), (5, -1), (5, 5), (5, 6), (True, 0), (5, False), (5.5, 0), (5, 1.5)],
)
def test_invalid_parameters(size, overlap):
    with pytest.raises(ValueError):
        chunk_document(document("text"), size, overlap)


def test_readable_boundary():
    chunks = chunk_document(document("one two three four five six"), 15, 3)
    assert chunks[0].text == "one two three "


def test_metadata_and_pages():
    source = Document("original.pdf", (DocumentPart("page one words", 1), DocumentPart("next", 3)))
    chunks = chunk_document(source, 8, 2, document_id=2)
    assert all(chunk.filename == "original.pdf" for chunk in chunks)
    assert {chunk.page for chunk in chunks} == {1, 3}
    assert len({chunk.identifier for chunk in chunks}) == len(chunks)
    assert chunks[0].identifier == "doc-0002-chunk-0001"
    assert chunks[-1].start == 0


def test_whitespace_only_part_is_skipped():
    assert chunk_document(document(" \n\t")) == []


@pytest.mark.parametrize(
    "text",
    [
        "First sentence. Second sentence! Third question? End.",
        "Heading\n\nParagraph one.\nWrapped PDF line continues here.\n\nNext paragraph.",
        "# Title\n- First bullet\n- Second bullet\n1. Numbered item\n## Next heading\nText",
        "a" * 130,
        "",
        " \n",
        "Exact size.",
    ],
)
@pytest.mark.parametrize("size,overlap", [(1, 0), (12, 0), (25, 8), (25, 24)])
def test_sentence_source_coverage(text, size, overlap):
    chunks = chunk_document(document(text), size, overlap, strategy="Sentence-aware")
    assert chunks == chunk_document(document(text), size, overlap, strategy="Sentence-aware")
    if not text.strip():
        assert not chunks
        return
    assert chunks[0].start == 0
    assert chunks[-1].end == len(text)
    covered = set()
    for chunk in chunks:
        assert 0 < len(chunk.text) <= size
        assert chunk.text == text[chunk.start : chunk.end]
        covered.update(range(chunk.start, chunk.end))
    assert covered == set(range(len(text)))
    for previous, current in zip(chunks, chunks[1:], strict=False):
        assert previous.start < current.start <= previous.end
        assert 0 <= previous.end - current.start <= overlap


def test_sentence_boundaries_and_overlap():
    chunks = chunk_document(document("One. Two. Three. Four."), 16, 8, strategy="Sentence-aware")
    assert [chunk.text for chunk in chunks] == ["One. Two. ", "Two. Three. ", "Three. Four."]


def test_sentence_pages_and_ids():
    source = Document("source.pdf", (DocumentPart("One. Two. Three.", 2), DocumentPart("End.", 4)))
    chunks = chunk_document(source, 10, 3, 7, "Sentence-aware")
    assert all(chunk.filename == "source.pdf" for chunk in chunks)
    assert chunks[-1].page == 4
    assert chunks[-1].start == 0
    assert chunks[0].identifier == "doc-0007-chunk-0001"


def test_unknown_strategy():
    with pytest.raises(ValueError, match="chunking"):
        chunk_document(document("text"), strategy="unknown")


def test_sentence_overlap_does_not_force_splitting_a_fitting_sentence():
    chunks = chunk_document(
        document("One. Two. A longer sentence."), 20, 8, strategy="Sentence-aware"
    )
    assert [chunk.text for chunk in chunks] == ["One. Two. ", "A longer sentence."]


def test_high_sentence_overlap_still_prefers_sentence_boundaries():
    chunks = chunk_document(document("One. Two. Three."), 10, 9, strategy="Sentence-aware")
    assert [chunk.text for chunk in chunks] == ["One. Two. ", "Three."]
