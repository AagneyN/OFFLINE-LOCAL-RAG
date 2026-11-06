from backend.retriever import retrieve_documents

question = "Does RAG require fine-tuning?"

results = retrieve_documents(question, top_k=5)

print("\nTOP 5 RETRIEVAL RESULTS")
print("=" * 70)

for rank, result in enumerate(results, start=1):
    metadata = result["metadata"]

    print(
        f"Rank {rank}: "
        f"{metadata['document_id']} | "
        f"Chunk {metadata['chunk_index']} | "
        f"Distance: {result['distance']:.4f}"
    )
