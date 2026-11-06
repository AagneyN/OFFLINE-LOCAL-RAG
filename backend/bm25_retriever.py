import re

from rank_bm25 import BM25Okapi

from backend.vector_store import get_collection


def tokenize(text: str) -> list[str]:
    """Convert text into lowercase word tokens."""
    return re.findall(r"\b\w+\b", text.lower())


def retrieve_bm25(query: str, top_k: int = 3) -> list[dict]:
    """Retrieve document chunks using BM25 keyword relevance."""

    if not query.strip():
        return []

    collection = get_collection()

    records = collection.get(
        include=["documents", "metadatas"]
    )

    documents = records["documents"]
    metadatas = records["metadatas"]
    ids = records["ids"]

    if not documents:
        return []

    tokenized_documents = [
        tokenize(document)
        for document in documents
    ]

    bm25 = BM25Okapi(tokenized_documents)

    query_tokens = tokenize(query)
    scores = bm25.get_scores(query_tokens)

    ranked_indices = sorted(
        range(len(scores)),
        key=lambda index: scores[index],
        reverse=True
    )

    results = []

    for index in ranked_indices[:top_k]:
        results.append({
            "id": ids[index],
            "text": documents[index],
            "metadata": metadatas[index],
            "score": float(scores[index])
        })

    return results
