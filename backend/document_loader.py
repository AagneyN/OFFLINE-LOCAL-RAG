from pathlib import Path
import pymupdf


def load_txt(path: Path) -> str:
    """Extract text from a TXT file."""

    return path.read_text(encoding="utf-8-sig")


def load_pdf(path: Path) -> str:
    """Extract text from a PDF file."""

    with pymupdf.open(path) as pdf:
        pages = [
            page.get_text()
            for page in pdf
        ]

    return "\n".join(pages)


def load_document(path: Path) -> str:
    """Load a supported document based on its extension."""

    suffix = path.suffix.lower()

    if suffix == ".txt":
        return load_txt(path)

    elif suffix == ".pdf":
        return load_pdf(path)

    else:
        raise ValueError(
            f"Unsupported document format: {suffix}"
        )