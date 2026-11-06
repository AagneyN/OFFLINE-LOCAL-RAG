
from pathlib import Path

from backend.config import DB_DIR
from backend.rag_pipeline import ask_question
from backend.vector_store import get_collection_count


def main():
    """Run LocalRAG through an interactive command-line interface."""

    if not DB_DIR.exists():
        raise FileNotFoundError(
            "Vector database not found. Run index_documents.py first."
        )

    chunk_count = get_collection_count()

    if chunk_count == 0:
        raise RuntimeError(
            "The vector database is empty. Index documents first."
        )

    print(f"Loaded database with {chunk_count} chunks.")
    print("\nLocalRAG is ready.")
    print("Type 'exit', 'quit', or 'bye' to quit.\n")

    while True:
        question = input("You: ").strip()

        if question.lower() in {"exit", "quit", "bye"}:
            print("Goodbye!")
            break

        if not question:
            continue

        try:
            result = ask_question(question)

            print("\n--- Generated answer ---")
            print(result["answer"])

            if result["sources"]:
                print("\n--- Sources ---")

                for index, source in enumerate(
                    result["sources"], start=1
                ):
                    source_name = Path(
                        source["source"]
                    ).name

                    print(
                        f"[{index}] {source_name}"
                        f" | Chunk {source['chunk_index']}"
                        f" | Cosine distance: "
                        f"{source['distance']:.4f}"
                    )

            else:
                print("\nNo relevant sources found.")

            print("\n" + "-" * 60)

        except Exception as error:
            print(f"\nAn error occurred: {error}")
            print("You can try asking another question.\n")


if __name__ == "__main__":
    main()
