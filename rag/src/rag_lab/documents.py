"""Read local UTF-8 text and text-based PDFs without changing their contents."""

from dataclasses import dataclass
from pathlib import Path

import pymupdf


@dataclass(frozen=True)
class DocumentPart:
    text: str
    page: int | None = None


@dataclass(frozen=True)
class Document:
    filename: str
    parts: tuple[DocumentPart, ...]


def load_document(path: str | Path) -> Document:
    """Keep PDF pages separate; reject files with no extractable text (no OCR)."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in {".txt", ".md", ".pdf"}:
        raise ValueError(f"Unsupported file type: {path.name}. Use TXT, Markdown or PDF.")
    try:
        if suffix == ".pdf":
            with pymupdf.open(path) as pdf:
                if pdf.needs_pass:
                    raise ValueError(f"Password-protected PDF: {path.name}.")
                parts = tuple(
                    DocumentPart(page.get_text(), number)
                    for number, page in enumerate(pdf, start=1)
                )
        else:
            parts = (DocumentPart(path.read_text(encoding="utf-8-sig")),)
    except (OSError, UnicodeError, RuntimeError, pymupdf.FileDataError) as exc:
        raise ValueError(f"Could not read {path.name}. Check the file and its encoding.") from exc
    if not any(part.text.strip() for part in parts):
        raise ValueError(
            f"No text found in {path.name}. Scanned PDFs need OCR, which is not supported."
        )
    return Document(path.name, parts)
