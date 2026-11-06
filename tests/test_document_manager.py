
import pymupdf
import pytest

from backend import document_manager
from backend.document_manager import DocumentError


def make_pdf(text="", password=None) -> bytes:
    """Build a one-page PDF in memory (blank page if no text is given)."""
    document = pymupdf.open()
    page = document.new_page()

    if text:
        page.insert_textbox(pymupdf.Rect(50, 50, 500, 700), text)

    if password:
        data = document.tobytes(
            encryption=pymupdf.PDF_ENCRYPT_AES_256,
            user_pw=password,
            owner_pw=password
        )
    else:
        data = document.tobytes()

    document.close()
    return data


@pytest.fixture
def upload_dir(monkeypatch, tmp_path):
    docs = tmp_path / "docs"
    uploads = docs / "uploads"
    docs.mkdir()

    monkeypatch.setattr(document_manager, "DOCS_DIR", docs)
    monkeypatch.setattr(document_manager, "UPLOAD_DIR", uploads)

    return uploads


# ---------- sanitize_filename ----------

def test_sanitize_removes_directory_parts():
    assert document_manager.sanitize_filename("../../evil.txt") == "evil.txt"
    assert document_manager.sanitize_filename("C:\\temp\\notes.TXT") == "notes.txt"


def test_sanitize_replaces_invalid_characters():
    assert document_manager.sanitize_filename('a<b>c?.txt') == "a_b_c_.txt"


def test_sanitize_protects_windows_reserved_names():
    assert document_manager.sanitize_filename("nul.txt") == "_nul.txt"


def test_sanitize_rejects_empty_name():
    with pytest.raises(DocumentError):
        document_manager.sanitize_filename("  .txt")


# ---------- validate_document ----------

def test_validate_accepts_text_file():
    assert document_manager.validate_document(
        "notes.txt", b"A database stores data."
    ) == 1


def test_validate_accepts_pdf_with_text():
    chunks = document_manager.validate_document(
        "paper.pdf", make_pdf("Retrieval augmented generation basics.")
    )

    assert chunks >= 1


def test_validate_rejects_unsupported_type():
    with pytest.raises(DocumentError, match="Unsupported file type"):
        document_manager.validate_document("sheet.docx", b"data")


def test_validate_rejects_empty_file():
    with pytest.raises(DocumentError, match="empty"):
        document_manager.validate_document("empty.txt", b"")


def test_validate_rejects_whitespace_only_text():
    with pytest.raises(DocumentError, match="no text"):
        document_manager.validate_document("blank.txt", b"  \n\t \n")


def test_validate_rejects_non_utf8_text():
    with pytest.raises(DocumentError, match="UTF-8"):
        document_manager.validate_document(
            "latin.txt", "caf\xe9".encode("utf-16")
        )


def test_validate_rejects_corrupted_pdf():
    with pytest.raises(DocumentError, match="could not be read"):
        document_manager.validate_document(
            "broken.pdf", b"this is definitely not a pdf"
        )


def test_validate_rejects_pdf_without_text():
    with pytest.raises(DocumentError, match="No text could be extracted"):
        document_manager.validate_document("scan.pdf", make_pdf())


def test_validate_rejects_password_protected_pdf():
    with pytest.raises(DocumentError, match="password-protected"):
        document_manager.validate_document(
            "secret.pdf", make_pdf("Hidden text.", password="pw")
        )


# ---------- classify_upload / save_document ----------

def test_save_new_document(upload_dir):
    path = document_manager.save_document("notes.txt", b"First version.")

    assert path == upload_dir / "notes.txt"
    assert path.read_bytes() == b"First version."
    assert document_manager.classify_upload(
        "notes.txt", b"First version."
    ) == "identical"
    assert not list(upload_dir.glob("*.tmp"))


def test_classify_new_and_conflict(upload_dir):
    assert document_manager.classify_upload("notes.txt", b"x") == "new"

    document_manager.save_document("notes.txt", b"First version.")

    assert document_manager.classify_upload(
        "notes.txt", b"Second version."
    ) == "conflict"


def test_identical_upload_is_not_duplicated(upload_dir):
    document_manager.save_document("notes.txt", b"Same content.")
    document_manager.save_document("notes.txt", b"Same content.")

    assert [p.name for p in upload_dir.iterdir()] == ["notes.txt"]


def test_conflict_keep_both_never_overwrites(upload_dir):
    document_manager.save_document("notes.txt", b"Original.")
    second = document_manager.save_document("notes.txt", b"Changed.")
    third = document_manager.save_document("notes.txt", b"Changed again.")

    assert second.name == "notes (2).txt"
    assert third.name == "notes (3).txt"
    assert (upload_dir / "notes.txt").read_bytes() == b"Original."


def test_conflict_replace_updates_existing_file(upload_dir):
    document_manager.save_document("notes.txt", b"Original.")
    path = document_manager.save_document(
        "notes.txt", b"Updated.", on_conflict="replace"
    )

    assert path.name == "notes.txt"
    assert path.read_bytes() == b"Updated."
    assert [p.name for p in upload_dir.iterdir()] == ["notes.txt"]


def test_invalid_upload_is_not_saved(upload_dir):
    with pytest.raises(DocumentError):
        document_manager.save_document("broken.pdf", b"not a pdf")

    assert not upload_dir.exists() or not any(upload_dir.iterdir())


def test_path_traversal_name_stays_inside_upload_dir(upload_dir):
    path = document_manager.save_document("../../escape.txt", b"Contents.")

    assert path.parent == upload_dir
    assert path.name == "escape.txt"


def test_invalid_conflict_policy_is_rejected(upload_dir):
    with pytest.raises(ValueError):
        document_manager.save_document("a.txt", b"x", on_conflict="merge")


# ---------- list_documents ----------

def test_list_documents_combines_files_and_index(monkeypatch, upload_dir):
    docs = upload_dir.parent
    (docs / "indexed.txt").write_text("a", encoding="utf-8")
    (docs / "pending.pdf").write_bytes(b"x")
    (docs / "ignore.docx").write_bytes(b"x")

    monkeypatch.setattr(
        document_manager,
        "get_indexed_documents",
        lambda: {"indexed.txt": 4, "removed.txt": 2}
    )

    rows = {
        row["document_id"]: row
        for row in document_manager.list_documents()
    }

    assert set(rows) == {"indexed.txt", "pending.pdf", "removed.txt"}
    assert rows["indexed.txt"]["status"] == "Indexed"
    assert rows["indexed.txt"]["chunks"] == 4
    assert rows["pending.pdf"]["status"] == "Not indexed"
    assert rows["pending.pdf"]["chunks"] == 0
    assert rows["removed.txt"]["status"] == "File missing"
