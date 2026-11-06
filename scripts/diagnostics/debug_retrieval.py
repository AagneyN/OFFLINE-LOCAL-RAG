
from pathlib import Path

from backend.retriever import retrieve_documents

question = "Does RAG require fine-tuning?"

results = retrieve_documents(question)

print("\nQUESTION:", question)
print("=" * 70)

for rank, result in enumerate(results, start=1):
    source = Path(result["metadata"]["source"]).name
    chunk_index = result["metadata"]["chunk_index"]
    distance = result["distance"]

    print(
        f"Rank {rank}: {source} | "
        f"Chunk {chunk_index} | "
        f"Distance: {distance:.4f}"
    )
