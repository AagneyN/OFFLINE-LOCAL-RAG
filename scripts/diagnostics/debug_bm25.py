from backend.bm25_retriever import retrieve_bm25

question = "Does RAG require fine-tuning?"

results = retrieve_bm25(question, top_k=5)

print("\nBM25 RETRIEVAL RESULTS")
print("=" * 70)

for rank, result in enumerate(results, start=1):
    metadata = result["metadata"]

    print(
        f"Rank {rank}: "
        f"{metadata['document_id']} | "
        f"Chunk {metadata['chunk_index']} | "
        f"BM25 Score: {result['score']:.4f}"
    )

    print("Content:", result["text"][:250])
    print("-" * 70)
