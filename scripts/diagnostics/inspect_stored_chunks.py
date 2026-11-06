from backend.vector_store import get_collection

collection = get_collection()

results = collection.get(
    where={
        "$and": [
            {"document_id": {"$eq": "rag_basics.txt"}},
            {"chunk_index": {"$eq": 0}}
        ]
    },
    include=["documents", "metadatas"]
)

print("\nSTORED CHUNK CHECK")
print("=" * 70)

if not results["ids"]:
    print("Expected chunk was not found in ChromaDB.")

else:
    for index, document in enumerate(results["documents"]):
        print("ID:", results["ids"][index])
        print("METADATA:", results["metadatas"][index])
        print("\nCONTENT:")
        print(document)
        print("\n" + "=" * 70)
