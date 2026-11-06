from backend.hybrid_retriever import retrieve_hybrid

question = "Does RAG require fine-tuning?"

results = retrieve_hybrid(question, top_k=5)

print("\nHYBRID RETRIEVAL RESULTS")
print("=" * 80)

for rank, result in enumerate(results, start=1):
    metadata = result["metadata"]

    print(f"\nRank: {rank}")
    print(f"Document: {metadata['document_id']}")
    print(f"Chunk: {metadata['chunk_index']}")
    print(f"RRF Score: {result['rrf_score']:.6f}")
    print(f"Semantic Rank: {result['semantic_rank']}")
    print(f"BM25 Rank: {result['bm25_rank']}")

print("\n" + "=" * 80)
