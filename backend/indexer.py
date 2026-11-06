
import hashlib

from backend.config import (
    DOCS_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    EMBEDDING_MODEL,
    SUPPORTED_EXTENSIONS
)

from backend.document_loader import load_document
from backend.chunker import split_into_chunks
from backend.embeddings import generate_embeddings
from backend.vector_store import get_collection


def generate_document_hash(text: str) -> str:
    """Generate a fingerprint for a document and its indexing settings."""
    signature = (
        text
        + f"|chunk_size={CHUNK_SIZE}"
        + f"|chunk_overlap={CHUNK_OVERLAP}"
        + f"|embedding_model={EMBEDDING_MODEL}"
    )

    return hashlib.sha256(
        signature.encode("utf-8")
    ).hexdigest()


def index_documents(
    progress_callback=None,
    continue_on_error: bool = False
) -> dict:
    """Index new and changed documents while preserving old data on failure.

    progress_callback(position, total, document_id) is called before each
    document is processed.

    By default the first failing document aborts the run. With
    continue_on_error=True the failure is recorded in report["failed"]
    instead, the document's previously indexed records are left untouched,
    and the remaining documents are still processed.
    """

    if not DOCS_DIR.exists():
        raise FileNotFoundError(
            f"Documents directory not found: {DOCS_DIR}"
        )

    files = sorted(
        path
        for path in DOCS_DIR.rglob("*")
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    if not files:
        raise RuntimeError(
            "No supported documents found. "
            "Indexing stopped to protect the database."
        )

    collection = get_collection()

    active_document_ids = set()
    indexed_count = 0
    skipped_count = 0
    deleted_count = 0
    total_new_chunks = 0
    failed = []

    for position, path in enumerate(files, start=1):
        document_id = path.relative_to(
            DOCS_DIR
        ).as_posix()

        active_document_ids.add(document_id)

        if progress_callback:
            progress_callback(position, len(files), document_id)

        try:
            text = load_document(path)
            chunks = split_into_chunks(text)

            existing = collection.get(
                where={"document_id": document_id},
                include=["metadatas"]
            )

            existing_ids = existing["ids"]
            existing_metadata = existing["metadatas"]

            if not chunks:
                if existing_ids:
                    collection.delete(ids=existing_ids)

                print(f"Skipped empty document: {document_id}")
                continue

            document_hash = generate_document_hash(text)

            # Use versioned IDs so old records remain untouched
            # until the new version has been inserted successfully.
            version = document_hash[:16]

            new_ids = [
                f"{document_id}::v_{version}::chunk_{i}"
                for i in range(len(chunks))
            ]

            existing_new_version = {
                record_id
                for record_id, metadata in zip(
                    existing_ids, existing_metadata
                )
                if metadata.get("document_hash") == document_hash
            }

            # Recover safely if the previous run inserted the new
            # version but failed while cleaning up the old version.
            if set(new_ids).issubset(existing_new_version):
                obsolete_ids = [
                    record_id
                    for record_id in existing_ids
                    if record_id not in new_ids
                ]

                if obsolete_ids:
                    collection.delete(ids=obsolete_ids)

                print(f"Unchanged, skipped: {document_id}")
                skipped_count += 1
                continue

            if any(
                metadata.get("document_hash") == document_hash
                for metadata in existing_metadata
            ):
                print(
                    f"Completing previous indexing operation: "
                    f"{document_id}"
                )

            print(f"Indexing: {document_id}")

            # Prepare everything before touching existing records.
            embeddings = generate_embeddings(chunks)

            if len(embeddings) != len(chunks):
                raise RuntimeError(
                    f"Embedding count mismatch for {document_id}"
                )

            metadatas = [
                {
                    "document_id": document_id,
                    "source": str(path.resolve()),
                    "chunk_index": i,
                    "document_hash": document_hash
                }
                for i in range(len(chunks))
            ]

            # Insert the new version first.
            collection.upsert(
                ids=new_ids,
                documents=chunks,
                embeddings=embeddings,
                metadatas=metadatas
            )

            # Delete old versions only after successful insertion.
            obsolete_ids = [
                record_id
                for record_id in existing_ids
                if record_id not in new_ids
            ]

            if obsolete_ids:
                collection.delete(ids=obsolete_ids)

            indexed_count += 1
            total_new_chunks += len(chunks)

            print(
                f"Indexed {document_id}: "
                f"{len(chunks)} chunks"
            )

        except Exception as error:
            if not continue_on_error:
                raise

            failed.append({
                "document_id": document_id,
                "error": str(error) or type(error).__name__
            })

            print(f"Failed to index {document_id}: {error}")

    # Remove records belonging to documents that no longer exist.
    all_records = collection.get(
        include=["metadatas"]
    )

    existing_document_ids = {
        metadata.get("document_id")
        for metadata in all_records["metadatas"]
        if metadata.get("document_id")
    }

    for document_id in existing_document_ids:
        if document_id not in active_document_ids:
            collection.delete(
                where={"document_id": document_id}
            )

            print(f"Removed deleted document: {document_id}")
            deleted_count += 1

    report = {
        "documents_found": len(files),
        "documents_indexed": indexed_count,
        "documents_unchanged": skipped_count,
        "documents_removed": deleted_count,
        "documents_failed": len(failed),
        "failed": failed,
        "new_chunks": total_new_chunks,
        "total_chunks": collection.count()
    }

    print("\n========== INDEXING REPORT ==========")

    for key, value in report.items():
        print(f"{key}: {value}")

    return report
