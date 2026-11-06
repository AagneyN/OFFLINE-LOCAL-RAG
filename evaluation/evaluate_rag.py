
from pathlib import Path

from backend.config import MAX_DISTANCE, TOP_K
from backend.retriever import retrieve_documents

from .evaluation_dataset import EVALUATION_DATASET


def evaluate():
    """Evaluate retrieval quality and rejection behaviour."""

    total_precision = 0
    total_recall = 0
    answerable_count = 0

    correct_rejections = 0
    total_rejection_tests = 0

    print("\n========== LOCAL RAG EVALUATION ==========\n")
    print(f"Evaluation questions: {len(EVALUATION_DATASET)}")
    print(f"Top-K: {TOP_K}")
    print(f"Maximum distance: {MAX_DISTANCE}\n")

    for test in EVALUATION_DATASET:
        question = test["question"]
        expected_chunks = set(test["expected_chunks"])
        expected_answerable = test["answerable"]

        # Retrieve chunks using the modular backend.
        raw_results = retrieve_documents(question)

        # Apply the configured relevance threshold.
        retrieved = [
            item
            for item in raw_results
            if item["distance"] <= MAX_DISTANCE
        ]

        retrieved_chunks = {
            (
                Path(item["metadata"]["source"]).name,
                item["metadata"]["chunk_index"]
            )
            for item in retrieved
        }

        relevant_found = len(
            retrieved_chunks.intersection(expected_chunks)
        )

        best_distance = (
            raw_results[0]["distance"]
            if raw_results
            else float("inf")
        )

        predicted_answerable = (
            bool(raw_results)
            and best_distance <= MAX_DISTANCE
        )

        # Calculate retrieval metrics for answerable questions.
        if expected_answerable:
            precision = (
                relevant_found / len(retrieved_chunks)
                if retrieved_chunks
                else 0
            )

            recall = (
                relevant_found / len(expected_chunks)
                if expected_chunks
                else 0
            )

            total_precision += precision
            total_recall += recall
            answerable_count += 1

        # Calculate rejection accuracy for all questions.
        if predicted_answerable == expected_answerable:
            correct_rejections += 1

        total_rejection_tests += 1

        status = (
            "ACCEPTED"
            if predicted_answerable
            else "REJECTED"
        )

        print(f"Question: {question}")
        print(f"Expected answerable: {expected_answerable}")
        print(f"Predicted: {status}")
        print(f"Best distance: {best_distance:.4f}")

        print(f"Expected chunks: {sorted(expected_chunks)}")
        print(f"Retrieved chunks: {sorted(retrieved_chunks)}")
        print(f"Relevant chunks found: {relevant_found}")

        for item in retrieved:
            source_name = Path(
                item["metadata"]["source"]
            ).name

            print(
                f"  - {source_name}"
                f" | Chunk {item['metadata']['chunk_index']}"
                f" | Distance: {item['distance']:.4f}"
            )

        print("-" * 60)

    # Calculate macro-average metrics.
    precision_at_k = (
        total_precision / answerable_count
        if answerable_count
        else 0
    )

    recall_at_k = (
        total_recall / answerable_count
        if answerable_count
        else 0
    )

    rejection_accuracy = (
        correct_rejections / total_rejection_tests
        if total_rejection_tests
        else 0
    )

    print("\n========== FINAL RESULTS ==========")
    print(f"Total questions: {len(EVALUATION_DATASET)}")
    print(f"Answerable questions: {answerable_count}")
    print(
        f"Unanswerable questions: "
        f"{total_rejection_tests - answerable_count}"
    )
    print(f"Precision@K: {precision_at_k:.2%}")
    print(f"Recall@K: {recall_at_k:.2%}")
    print(f"Rejection accuracy: {rejection_accuracy:.2%}")
    print("\nEvaluation completed.")


if __name__ == "__main__":
    evaluate()
