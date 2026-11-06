
from backend.config import DOCS_DIR, SUPPORTED_EXTENSIONS
from backend.document_loader import load_document
from backend.chunker import split_into_chunks


def inspect_all_chunks():
    """Display all document chunks using the backend configuration."""

    files = sorted(
        path
        for path in DOCS_DIR.rglob("*")
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    if not files:
        print("No supported documents found.")
        return

    total_chunks = 0

    for path in files:
        document_id = path.relative_to(
            DOCS_DIR
        ).as_posix()

        text = load_document(path)
        chunks = split_into_chunks(text)

        total_chunks += len(chunks)

        print("\n" + "=" * 70)
        print(f"DOCUMENT: {document_id}")
        print(f"TOTAL CHUNKS: {len(chunks)}")
        print("=" * 70)

        for index, chunk in enumerate(chunks):
            print(f"\nCHUNK {index}:")
            print(f"Word Count: {len(chunk.split())}")
            print("-" * 70)
            print(chunk)

    print("\n" + "=" * 70)
    print(f"TOTAL DOCUMENTS: {len(files)}")
    print(f"TOTAL CHUNKS: {total_chunks}")
    print("=" * 70)


if __name__ == "__main__":
    inspect_all_chunks()
