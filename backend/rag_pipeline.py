
from backend.hybrid_retriever import retrieve_hybrid
from backend.generator import REFUSAL_MESSAGE, generate_answer
from backend.vector_store import get_chunk
from backend.config import ADJACENT_CHUNKS, MAX_DISTANCE, TOP_K


def get_adjacent_chunks(results: list[dict], count: int = ADJACENT_CHUNKS) -> list[dict]:
    """The chunk that follows each of the best `count` results.

    Chunks are cut at a fixed word count, so an answer can run across the end
    of one chunk into the next. Chunks already in `results` are not repeated.
    """

    def key(result):
        return (
            result["metadata"].get("document_id"),
            result["metadata"].get("chunk_index")
        )

    present = {key(result) for result in results}
    adjacent = []

    for result in results[:count]:
        document_id, chunk_index = key(result)

        if document_id is None or chunk_index is None:
            continue

        following = (document_id, chunk_index + 1)

        if following in present:
            continue

        chunk = get_chunk(*following)

        if chunk:
            present.add(following)
            adjacent.append({
                "document_id": document_id,
                "chunk_index": chunk_index + 1,
                "text": chunk["text"]
            })

    return adjacent


def ask_question(question: str) -> dict:
    """Answer a question using hybrid retrieval and local generation."""

    if not question.strip():
        return {
            "answer": "Please enter a question.",
            "sources": [],
            "adjacent": [],
            "declined": False,
            "retrieved_count": 0
        }

    # Retrieve documents using hybrid search.
    results = retrieve_hybrid(
        question,
        top_k=TOP_K
    )

    # Apply the existing semantic relevance threshold.
    relevant_results = [
        result for result in results
        if result["distance"] is not None
        and result["distance"] <= MAX_DISTANCE
    ]

    if not relevant_results:
        return {
            "answer": (
                "I could not find sufficient relevant information "
                "in the knowledge base."
            ),
            "sources": [],
            "adjacent": [],
            "declined": False,
            "retrieved_count": 0
        }

    context_parts = []
    sources = []

    for index, result in enumerate(relevant_results, start=1):
        context_parts.append(
            f"[{index}] {result['text']}"
        )

        sources.append({
            "source": result["metadata"].get(
                "source",
                result["metadata"].get("document_id", "Unknown")
            ),
            "document_id": result["metadata"].get(
                "document_id",
                result["metadata"].get("source", "Unknown")
            ),
            "chunk_index": result["metadata"].get("chunk_index"),
            "distance": result["distance"],
            "rrf_score": result["rrf_score"],
            "bm25_score": result.get("bm25_score"),
            "text": result["text"]
        })

    # Passages that continue the best matches. They go to the model as extra
    # context but are not search results, so they are listed separately.
    adjacent = get_adjacent_chunks(relevant_results)

    for offset, chunk in enumerate(adjacent, start=len(sources) + 1):
        context_parts.append(f"[{offset}] {chunk['text']}")

    context = "\n\n".join(context_parts)

    answer = generate_answer(question, context)

    return {
        "answer": answer,
        "sources": sources,
        "adjacent": adjacent,
        "declined": answer == REFUSAL_MESSAGE,
        "retrieved_count": len(relevant_results)
    }
