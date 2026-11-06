

from backend import retriever


class FakeCollection:
    def __init__(self, count=2):
        self._count = count

    def count(self):
        return self._count

    def query(self, **kwargs):
        return {
            "documents": [["Database fundamentals", "SQL basics"]],
            "metadatas": [[
                {"source": "databases_sql.txt", "chunk_index": 0},
                {"source": "databases_sql.txt", "chunk_index": 1}
            ]],
            "distances": [[0.2, 0.4]]
        }


def test_empty_query_returns_empty_list():
    result = retriever.retrieve_documents("   ")

    assert result == []


def test_empty_collection_returns_empty_list(monkeypatch):
    monkeypatch.setattr(
        retriever,
        "get_collection",
        lambda: FakeCollection(count=0)
    )

    result = retriever.retrieve_documents("What is SQL?")

    assert result == []


def test_retrieval_returns_expected_results(monkeypatch):
    monkeypatch.setattr(
        retriever,
        "get_collection",
        lambda: FakeCollection()
    )

    monkeypatch.setattr(
        retriever,
        "generate_embedding",
        lambda query: [0.1] * 384
    )

    result = retriever.retrieve_documents("What is SQL?")

    assert len(result) == 2
    assert result[0]["text"] == "Database fundamentals"
    assert result[0]["distance"] == 0.2
    assert result[1]["text"] == "SQL basics"

def test_retrieval_uses_top_k_limit(monkeypatch):
    captured = {}

    class TrackingCollection(FakeCollection):
        def query(self, **kwargs):
            captured.update(kwargs)
            return super().query(**kwargs)

    monkeypatch.setattr(
        retriever,
        "get_collection",
        lambda: TrackingCollection(count=2)
    )

    monkeypatch.setattr(
        retriever,
        "generate_embedding",
        lambda query: [0.1] * 384
    )

    retriever.retrieve_documents("What is SQL?", top_k=1)

    assert captured["n_results"] == 1
