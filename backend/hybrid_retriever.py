
from backend.retriever import retrieve_documents
from backend.bm25_retriever import retrieve_bm25
from backend.vector_store import get_collection


RRF_K = 60


def get_record_key(result: dict) -> tuple:
    """Create a consistent identifier for both retrieval methods."""
    metadata = result["metadata"]

    return (
        metadata["document_id"],
        metadata["chunk_index"]
    )


def retrieve_hybrid(query: str, top_k: int = 5) -> list[dict]:
    """Combine semantic and BM25 rankings using RRF."""

    if not query.strip():
        return []

    total_chunks = get_collection().count()

    if total_chunks == 0:
        return []

    semantic_results = retrieve_documents(
        query,
        top_k=total_chunks
    )

    bm25_results = retrieve_bm25(
        query,
        top_k=total_chunks
    )

    fused_results = {}

    def add_ranking(results, method):
        for rank, result in enumerate(results, start=1):
            record_key = get_record_key(result)

            if record_key not in fused_results:
                fused_results[record_key] = {
                    "id": result.get("id"),
                    "text": result["text"],
                    "metadata": result["metadata"],
                    "rrf_score": 0.0,
                    "semantic_rank": None,
                    "bm25_rank": None,
                    "distance": None,
                    "bm25_score": None
                }

            item = fused_results[record_key]

            item["rrf_score"] += 1.0 / (RRF_K + rank)
            item[f"{method}_rank"] = rank

            if method == "semantic":
                item["distance"] = result["distance"]

            if method == "bm25":
                item["bm25_score"] = result["score"]

    add_ranking(semantic_results, "semantic")

    positive_bm25_results = [
        result for result in bm25_results
        if result["score"] > 0
    ]

    add_ranking(positive_bm25_results, "bm25")

    ranked_results = sorted(
        fused_results.values(),
        key=lambda item: item["rrf_score"],
        reverse=True
    )

    return ranked_results[:top_k]
