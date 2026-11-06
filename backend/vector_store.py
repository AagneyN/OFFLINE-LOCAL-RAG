from collections import Counter

import chromadb
from chromadb.config import Settings

from backend.config import (
    DB_DIR,
    COLLECTION_NAME,
    DISTANCE_METRIC
)


# Initialize persistent ChromaDB client.
# Telemetry is switched off (it is on by default) so that LocalRAG makes
# no outbound network calls.
_client = chromadb.PersistentClient(
    path=str(DB_DIR),
    settings=Settings(anonymized_telemetry=False)
)


def get_collection():
    """
    Get the existing collection or create it if necessary.
    """

    collection = _client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={
            "hnsw:space": DISTANCE_METRIC
        }
    )

    return collection


def get_collection_count() -> int:
    """Return the number of stored chunks."""

    collection = get_collection()

    return collection.count()


def get_chunk(document_id: str, chunk_index: int) -> dict | None:
    """Return one stored chunk of the current document version, or None."""

    records = get_collection().get(
        where={
            "$and": [
                {"document_id": document_id},
                {"chunk_index": chunk_index}
            ]
        },
        include=["documents", "metadatas"]
    )

    if not records["ids"]:
        return None

    return {
        "text": records["documents"][0],
        "metadata": records["metadatas"][0]
    }


def get_indexed_documents() -> dict[str, int]:
    """Return {document_id: chunk_count} for every indexed document."""

    records = get_collection().get(include=["metadatas"])

    return dict(
        Counter(
            metadata["document_id"]
            for metadata in records["metadatas"]
            if metadata.get("document_id")
        )
    )