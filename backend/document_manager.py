"""Validation, storage and listing of user-uploaded documents."""

import gc
import os
import re
import tempfile
from pathlib import Path

from backend.config import DOCS_DIR, UPLOAD_DIR, SUPPORTED_EXTENSIONS
from backend.chunker import split_into_chunks
from backend.document_loader import load_document
from backend.vector_store import get_indexed_documents


class DocumentError(ValueError):
    """An uploaded document cannot be used. The message is user-readable."""


_INVALID_CHARACTERS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_RESERVED_NAMES = re.compile(r"(con|prn|aux|nul|com\d|lpt\d)", re.IGNORECASE)


def sanitize_filename(name: str) -> str:
    """Reduce an uploaded file name to a safe name for a single folder."""

    # Drop any directory part so a name can never escape the upload folder.
    base = re.split(r"[\\/]", name)[-1].strip()
    base = _INVALID_CHARACTERS.sub("_", base)

    path = Path(base)
    stem = path.stem.strip(" .")

    if not stem or not path.suffix:
        raise DocumentError(
            "The file needs a name and a .txt or .pdf extension."
        )

    if _RESERVED_NAMES.fullmatch(stem):
        stem = f"_{stem}"

    return f"{stem[:120]}{path.suffix.lower()}"


def validate_document(name: str, data: bytes) -> int:
    """Check that a file can be loaded and chunked. Returns the chunk count."""

    suffix = Path(name).suffix.lower()

    if suffix not in SUPPORTED_EXTENSIONS:
        raise DocumentError(
            f"Unsupported file type '{suffix or name}'. "
            "Only TXT and PDF files can be indexed."
        )

    if not data:
        raise DocumentError("The file is empty.")

    # Run the real loader on a throw-away copy so that a broken file is
    # rejected before it ever reaches the document library.
    #
    # The error is only recorded inside this block and raised after it:
    # on Windows a PDF that failed to open stays locked until the exception
    # (and the half-built document it references) is garbage-collected, and
    # a locked file would make the temporary folder's cleanup crash.
    text = ""
    failure = None

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as folder:
        path = Path(folder) / f"upload{suffix}"
        path.write_bytes(data)

        try:
            text = load_document(path)

        except UnicodeDecodeError:
            failure = (
                "The text file is not valid UTF-8. "
                "Re-save it as UTF-8 and upload it again."
            )

        except Exception as error:
            failure = (
                "The PDF could not be read. It may be corrupted "
                "or password-protected."
                if suffix == ".pdf"
                else f"The file could not be read: {error}"
            )

        gc.collect()

    if failure:
        raise DocumentError(failure)

    chunks = split_into_chunks(text)

    if not chunks:
        if suffix == ".pdf":
            raise DocumentError(
                "No text could be extracted from this PDF. Scanned or "
                "image-only PDFs are not supported (there is no OCR)."
            )

        raise DocumentError("The file contains no text.")

    return len(chunks)


def classify_upload(name: str, data: bytes) -> str:
    """Compare an upload with the library: 'new', 'identical' or 'conflict'."""

    target = UPLOAD_DIR / sanitize_filename(name)

    if not target.exists():
        return "new"

    return "identical" if target.read_bytes() == data else "conflict"


def _unused_name(name: str) -> Path:
    """Return a path in the upload folder that does not exist yet."""

    path = Path(name)
    candidate = UPLOAD_DIR / name
    number = 2

    while candidate.exists():
        candidate = UPLOAD_DIR / f"{path.stem} ({number}){path.suffix}"
        number += 1

    return candidate


def save_document(
    name: str,
    data: bytes,
    on_conflict: str = "keep_both"
) -> Path:
    """Validate an upload and store it in the upload folder.

    If a different file with the same name already exists, on_conflict
    decides what happens:
      "keep_both" (default) - save the new file under a numbered name.
      "replace"             - replace the existing file (an update). The
                              indexer only drops the old chunks after the
                              new version has been indexed successfully.
    Identical content is never written twice.
    """

    if on_conflict not in {"keep_both", "replace"}:
        raise ValueError(f"Unknown conflict policy: {on_conflict}")

    validate_document(name, data)

    safe_name = sanitize_filename(name)
    target = UPLOAD_DIR / safe_name

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    if target.exists():
        if target.read_bytes() == data:
            return target

        if on_conflict == "keep_both":
            target = _unused_name(safe_name)

    # Write beside the target and swap it in, so a crash never leaves
    # a half-written document behind.
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_bytes(data)
    os.replace(temporary, target)

    return target


def list_documents() -> list[dict]:
    """List library files together with their indexing status."""

    indexed = get_indexed_documents()
    documents = []
    seen = set()

    if DOCS_DIR.exists():
        for path in sorted(DOCS_DIR.rglob("*")):
            if (
                not path.is_file()
                or path.suffix.lower() not in SUPPORTED_EXTENSIONS
            ):
                continue

            document_id = path.relative_to(DOCS_DIR).as_posix()
            seen.add(document_id)

            documents.append({
                "document_id": document_id,
                "type": path.suffix.lower().lstrip(".").upper(),
                "size_kb": round(path.stat().st_size / 1024, 1),
                "chunks": indexed.get(document_id, 0),
                "status": (
                    "Indexed" if document_id in indexed
                    else "Not indexed"
                )
            })

    # Indexed documents whose file has been removed; the next indexing
    # run cleans these up.
    for document_id, chunks in sorted(indexed.items()):
        if document_id not in seen:
            documents.append({
                "document_id": document_id,
                "type": Path(document_id).suffix.lstrip(".").upper(),
                "size_kb": None,
                "chunks": chunks,
                "status": "File missing"
            })

    return documents
