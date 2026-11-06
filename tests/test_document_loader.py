
import pymupdf
import pytest

from backend.document_loader import load_document


def test_load_txt_file(tmp_path):
    file_path = tmp_path / "sample.txt"
    file_path.write_text("This is a test document.", encoding="utf-8")

    result = load_document(file_path)

    assert result == "This is a test document."


def test_load_txt_with_bom(tmp_path):
    file_path = tmp_path / "sample.txt"
    file_path.write_text(
        "\ufeffThis is a test document.",
        encoding="utf-8"
    )

    result = load_document(file_path)

    assert result == "This is a test document."


def test_load_pdf_file(tmp_path):
    file_path = tmp_path / "sample.pdf"

    pdf = pymupdf.open()
    page = pdf.new_page()
    page.insert_text((50, 50), "This is a PDF test.")
    pdf.save(file_path)
    pdf.close()

    result = load_document(file_path)

    assert "This is a PDF test." in result


def test_unsupported_file_format(tmp_path):
    file_path = tmp_path / "sample.docx"
    file_path.write_text("Test content.", encoding="utf-8")

    with pytest.raises(ValueError):
        load_document(file_path)
