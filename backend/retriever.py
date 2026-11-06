from backend.embeddings import generate_embedding
from backend.vector_store import get_collection
from backend.config import TOP_K


def retrieve_documents(query: str, top_k: int = TOP_K) -> list[dict]:
    """
    Retrieve the most relevant document chunks for a query.
    """

    if not query.strip():
        return []

    collection = get_collection()

    if collection.count() == 0:
        return []

    query_embedding = generate_embedding(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"]
    )

    retrieved_documents = []

    for i in range(len(results["documents"][0])):
        retrieved_documents.append({
            "text": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i]
        })

    return retrieved_documents