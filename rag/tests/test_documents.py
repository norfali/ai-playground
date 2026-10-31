"""Local document parsing; generated PDF fixtures contain no personal data."""

import pymupdf
import pytest

from rag_lab.documents import load_document


@pytest.mark.parametrize("suffix", [".txt", ".md", ".TXT"])
def test_text_loading(tmp_path, suffix):
    path = tmp_path / f"notes{suffix}"
    path.write_text("Résumé\n\nOriginal spacing.\n", encoding="utf-8")
    document = load_document(path)
    assert document.filename == path.name
    assert document.parts[0].text == "Résumé\n\nOriginal spacing.\n"
    assert document.parts[0].page is None


def test_pdf_loading_preserves_pages(tmp_path):
    path = tmp_path / "notes.pdf"
    with pymupdf.open() as pdf:
        pdf.new_page().insert_text((72, 72), "First page")
        pdf.new_page()
        pdf.new_page().insert_text((72, 72), "Third page")
        pdf.save(path)
    document = load_document(path)
    assert [part.page for part in document.parts] == [1, 2, 3]
    assert document.parts[0].text.strip() == "First page"
    assert document.parts[1].text == ""
    assert document.parts[2].text.strip() == "Third page"


@pytest.mark.parametrize("content", ["", " \n\t"])
def test_empty_text(tmp_path, content):
    path = tmp_path / "empty.txt"
    path.write_text(content)
    with pytest.raises(ValueError, match="No text found"):
        load_document(path)


def test_empty_pdf(tmp_path):
    path = tmp_path / "scanned.pdf"
    with pymupdf.open() as pdf:
        pdf.new_page()
        pdf.save(path)
    with pytest.raises(ValueError, match="OCR"):
        load_document(path)


@pytest.mark.parametrize("suffix", [".pdf", ".txt", ".md"])
def test_malformed_file(tmp_path, suffix):
    path = tmp_path / f"broken{suffix}"
    path.write_bytes(b"\xff\xfe\x00not a PDF")
    with pytest.raises(ValueError, match="Could not read"):
        load_document(path)


def test_unsupported_type(tmp_path):
    with pytest.raises(ValueError, match="Unsupported file type"):
        load_document(tmp_path / "unsupported.docx")


def test_missing_file(tmp_path):
    with pytest.raises(ValueError, match="Could not read"):
        load_document(tmp_path / "missing.txt")


def test_encrypted_pdf(tmp_path):
    path = tmp_path / "locked.pdf"
    with pymupdf.open() as pdf:
        pdf.new_page().insert_text((72, 72), "Private")
        pdf.save(path, encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw="password")
    with pytest.raises(ValueError, match="Password-protected"):
        load_document(path)
