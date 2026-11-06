
import pytest

from backend import indexer


class FakeCollection:
    def __init__(self):
        self.records = {}

    def get(self, where=None, include=None):
        records = self.records

        if where:
            key, value = next(iter(where.items()))
            records = {
                record_id: record
                for record_id, record in records.items()
                if record["metadata"].get(key) == value
            }

        return {
            "ids": list(records.keys()),
            "metadatas": [
                record["metadata"] for record in records.values()
            ]
        }

    def upsert(self, ids, documents, embeddings, metadatas):
        for record_id, document, embedding, metadata in zip(
            ids, documents, embeddings, metadatas
        ):
            self.records[record_id] = {
                "document": document,
                "embedding": embedding,
                "metadata": metadata
            }

    def delete(self, ids=None, where=None):
        if ids is not None:
            for record_id in ids:
                self.records.pop(record_id, None)

        elif where:
            key, value = next(iter(where.items()))

            to_delete = [
                record_id
                for record_id, record in self.records.items()
                if record["metadata"].get(key) == value
            ]

            for record_id in to_delete:
                self.records.pop(record_id, None)

    def count(self):
        return len(self.records)


def prepare_indexer(monkeypatch, tmp_path, collection):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()

    document = docs_dir / "sample.txt"
    document.write_text("Updated document content.", encoding="utf-8")

    monkeypatch.setattr(indexer, "DOCS_DIR", docs_dir)
    monkeypatch.setattr(indexer, "get_collection", lambda: collection)
    monkeypatch.setattr(
        indexer,
        "load_document",
        lambda path: path.read_text(encoding="utf-8")
    )

    monkeypatch.setattr(
        indexer,
        "split_into_chunks",
        lambda text: [text]
    )

    return document


def test_embedding_failure_preserves_existing_records(
    monkeypatch, tmp_path
):
    collection = FakeCollection()

    collection.records["old_chunk"] = {
        "document": "Previous document content.",
        "embedding": [0.1, 0.2],
        "metadata": {
            "document_id": "sample.txt",
            "document_hash": "old_hash",
            "source": "sample.txt",
            "chunk_index": 0
        }
    }

    prepare_indexer(monkeypatch, tmp_path, collection)

    def fail_embeddings(chunks):
        raise RuntimeError("Simulated embedding failure")

    monkeypatch.setattr(
        indexer,
        "generate_embeddings",
        fail_embeddings
    )

    with pytest.raises(RuntimeError, match="Simulated embedding failure"):
        indexer.index_documents()

    assert "old_chunk" in collection.records
    assert collection.records["old_chunk"]["document"] == (
        "Previous document content."
    )


def test_successful_indexing_replaces_old_records(
    monkeypatch, tmp_path
):
    collection = FakeCollection()

    collection.records["old_chunk"] = {
        "document": "Previous document content.",
        "embedding": [0.1, 0.2],
        "metadata": {
            "document_id": "sample.txt",
            "document_hash": "old_hash",
            "source": "sample.txt",
            "chunk_index": 0
        }
    }

    prepare_indexer(monkeypatch, tmp_path, collection)

    monkeypatch.setattr(
        indexer,
        "generate_embeddings",
        lambda chunks: [[0.3, 0.4]]
    )

    report = indexer.index_documents()

    assert "old_chunk" not in collection.records
    assert collection.count() == 1
    assert report["documents_indexed"] == 1
    assert report["total_chunks"] == 1


def test_no_documents_does_not_modify_database(
    monkeypatch, tmp_path
):
    collection = FakeCollection()

    collection.records["existing_chunk"] = {
        "document": "Existing content.",
        "embedding": [0.1, 0.2],
        "metadata": {
            "document_id": "existing.txt",
            "document_hash": "existing_hash"
        }
    }

    empty_docs = tmp_path / "empty_docs"
    empty_docs.mkdir()

    monkeypatch.setattr(indexer, "DOCS_DIR", empty_docs)
    monkeypatch.setattr(indexer, "get_collection", lambda: collection)

    with pytest.raises(RuntimeError, match="No supported documents"):
        indexer.index_documents()

    assert collection.count() == 1
    assert "existing_chunk" in collection.records


def test_progress_callback_reports_each_document(monkeypatch, tmp_path):
    collection = FakeCollection()
    document = prepare_indexer(monkeypatch, tmp_path, collection)
    (document.parent / "second.txt").write_text("Another one.", encoding="utf-8")

    monkeypatch.setattr(
        indexer, "generate_embeddings", lambda chunks: [[0.3, 0.4]]
    )

    calls = []

    indexer.index_documents(
        progress_callback=lambda position, total, document_id: calls.append(
            (position, total, document_id)
        )
    )

    assert calls == [
        (1, 2, "sample.txt"),
        (2, 2, "second.txt")
    ]


def test_continue_on_error_isolates_failing_document(monkeypatch, tmp_path):
    collection = FakeCollection()

    collection.records["old_bad_chunk"] = {
        "document": "Previously indexed content.",
        "embedding": [0.1, 0.2],
        "metadata": {
            "document_id": "bad.txt",
            "document_hash": "old_hash",
            "source": "bad.txt",
            "chunk_index": 0
        }
    }

    document = prepare_indexer(monkeypatch, tmp_path, collection)
    (document.parent / "bad.txt").write_text("Corrupt.", encoding="utf-8")

    def load(path):
        if path.name == "bad.txt":
            raise ValueError("Cannot read this file")

        return path.read_text(encoding="utf-8")

    monkeypatch.setattr(indexer, "load_document", load)
    monkeypatch.setattr(
        indexer, "generate_embeddings", lambda chunks: [[0.3, 0.4]]
    )

    report = indexer.index_documents(continue_on_error=True)

    assert report["documents_indexed"] == 1
    assert report["documents_failed"] == 1
    assert report["failed"] == [
        {"document_id": "bad.txt", "error": "Cannot read this file"}
    ]
    # The healthy document was indexed and the failed one was not removed.
    assert any(r["metadata"]["document_id"] == "sample.txt"
               for r in collection.records.values())
    assert "old_bad_chunk" in collection.records


def test_failure_still_raises_by_default(monkeypatch, tmp_path):
    collection = FakeCollection()
    prepare_indexer(monkeypatch, tmp_path, collection)

    monkeypatch.setattr(
        indexer,
        "load_document",
        lambda path: (_ for _ in ()).throw(ValueError("unreadable"))
    )

    with pytest.raises(ValueError, match="unreadable"):
        indexer.index_documents()
