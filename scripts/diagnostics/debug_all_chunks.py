from pathlib import Path

from backend.embeddings import generate_embedding
from backend.vector_store import get_collection

question = "Does RAG require fine-tuning?"

collection = get_collection()
query_embedding = generate_embedding(question)

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=collection.count(),
    include=["documents", "metadatas", "distances"]
)

print("\nFULL RETRIEVAL RANKING")
print("=" * 80)
print("Question:", question)
print("Total chunks examined:", len(results["documents"][0]))
print("=" * 80)

for rank, (document, metadata, distance) in enumerate(
    zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ),
    start=1
):
    source = Path(metadata["source"]).name
    chunk_index = metadata["chunk_index"]

    marker = " <-- EXPECTED CHUNK" if (
        source == "rag_basics.txt" and chunk_index == 0
    ) else ""

    print(
        f"Rank {rank:02d} | {source:25} | "
        f"Chunk {chunk_index:02d} | "
        f"Distance: {distance:.4f}{marker}"
    )

print("\nLower distance means greater similarity.")
