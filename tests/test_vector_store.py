
from backend.vector_store import (
    get_collection,
    get_collection_count
)


def test_collection_is_created_or_accessible():
    collection = get_collection()

    assert collection is not None


def test_collection_has_correct_name():
    collection = get_collection()

    assert collection.name == "localrag"


def test_collection_count_is_valid():
    count = get_collection_count()

    assert isinstance(count, int)
    assert count >= 0


def test_collection_access_is_consistent():
    first = get_collection()
    second = get_collection()

    assert first.name == second.name
    assert first.count() == second.count()


def test_get_indexed_documents_counts_chunks_per_document(monkeypatch):
    from backend import vector_store

    class FakeCollection:
        def get(self, include=None):
            return {
                "metadatas": [
                    {"document_id": "a.txt"},
                    {"document_id": "a.txt"},
                    {"document_id": "b.pdf"},
                    {"chunk_index": 0}
                ]
            }

    monkeypatch.setattr(vector_store, "get_collection", FakeCollection)

    assert vector_store.get_indexed_documents() == {"a.txt": 2, "b.pdf": 1}


def test_telemetry_is_disabled():
    from backend import vector_store

    assert vector_store._client.get_settings().anonymized_telemetry is False
