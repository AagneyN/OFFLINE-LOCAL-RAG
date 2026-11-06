
import pytest

from backend import hybrid_retriever


def make_result(
    document_id,
    chunk_index,
    distance=None,
    score=None
):
    result = {
        "text": f"Content from {document_id}, chunk {chunk_index}",
        "metadata": {
            "document_id": document_id,
            "chunk_index": chunk_index
        }
    }

    if distance is not None:
        result["distance"] = distance

    if score is not None:
        result["score"] = score

    return result


def test_get_record_key():
    result = make_result("rag_basics.txt", 0)

    assert hybrid_retriever.get_record_key(result) == (
        "rag_basics.txt",
        0
    )


def test_empty_query_returns_empty_list():
    assert hybrid_retriever.retrieve_hybrid("   ") == []


def test_empty_collection_returns_empty_list(monkeypatch):
    class FakeCollection:
        def count(self):
            return 0

    monkeypatch.setattr(
        hybrid_retriever,
        "get_collection",
        lambda: FakeCollection()
    )

    assert hybrid_retriever.retrieve_hybrid("RAG") == []


def test_hybrid_combines_rankings(monkeypatch):
    class FakeCollection:
        def count(self):
            return 2

    semantic_results = [
        make_result("rag_basics.txt", 0, distance=0.4),
        make_result("rag_basics.txt", 1, distance=0.5)
    ]

    bm25_results = [
        make_result("rag_basics.txt", 1, score=5.0),
        make_result("rag_basics.txt", 0, score=3.0)
    ]

    monkeypatch.setattr(
        hybrid_retriever,
        "get_collection",
        lambda: FakeCollection()
    )

    monkeypatch.setattr(
        hybrid_retriever,
        "retrieve_documents",
        lambda query, top_k: semantic_results
    )

    monkeypatch.setattr(
        hybrid_retriever,
        "retrieve_bm25",
        lambda query, top_k: bm25_results
    )

    results = hybrid_retriever.retrieve_hybrid(
        "What is RAG?",
        top_k=2
    )

    assert len(results) == 2

    assert results[0]["metadata"]["chunk_index"] == 0
    assert results[1]["metadata"]["chunk_index"] == 1

    assert results[0]["semantic_rank"] == 1
    assert results[0]["bm25_rank"] == 2

    assert results[1]["semantic_rank"] == 2
    assert results[1]["bm25_rank"] == 1

    assert results[0]["distance"] == 0.4
    assert results[0]["bm25_score"] == 3.0


def test_zero_bm25_scores_are_not_fused(monkeypatch):
    class FakeCollection:
        def count(self):
            return 1

    semantic_results = [
        make_result("rag_basics.txt", 0, distance=0.4)
    ]

    bm25_results = [
        make_result("rag_basics.txt", 0, score=0.0)
    ]

    monkeypatch.setattr(
        hybrid_retriever,
        "get_collection",
        lambda: FakeCollection()
    )

    monkeypatch.setattr(
        hybrid_retriever,
        "retrieve_documents",
        lambda query, top_k: semantic_results
    )

    monkeypatch.setattr(
        hybrid_retriever,
        "retrieve_bm25",
        lambda query, top_k: bm25_results
    )

    results = hybrid_retriever.retrieve_hybrid("RAG")

    assert len(results) == 1
    assert results[0]["semantic_rank"] == 1
    assert results[0]["bm25_rank"] is None
    assert results[0]["bm25_score"] is None
