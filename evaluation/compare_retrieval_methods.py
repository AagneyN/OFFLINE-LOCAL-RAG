
from pathlib import Path

from backend.config import MAX_DISTANCE
from backend.retriever import retrieve_documents
from backend.bm25_retriever import retrieve_bm25
from backend.hybrid_retriever import retrieve_hybrid

from evaluation.evaluation_dataset import EVALUATION_DATASET


def get_chunk_key(result):
    metadata = result["metadata"]

    return (
        metadata["document_id"],
        metadata["chunk_index"]
    )


def evaluate_method(method, k):
    total_precision = 0
    total_recall = 0
    answerable_count = 0

    correct_rejections = 0
    total_questions = len(EVALUATION_DATASET)

    for test in EVALUATION_DATASET:
        question = test["question"]
        expected_chunks = test["expected_chunks"]
        answerable = test["answerable"]

        # Run the selected retrieval method.
        if method == "semantic":
            results = retrieve_documents(question, top_k=k)

            relevant_results = [
                result for result in results
                if result["distance"] <= MAX_DISTANCE
            ]

        elif method == "bm25":
            results = retrieve_bm25(question, top_k=k)

            relevant_results = [
                result for result in results
                if result["score"] > 0
            ]

        elif method == "hybrid":
            results = retrieve_hybrid(question, top_k=k)

            relevant_results = [
                result for result in results
                if result["distance"] is not None
                and result["distance"] <= MAX_DISTANCE
            ]

        else:
            raise ValueError(f"Unknown method: {method}")

        retrieved_chunks = {
            get_chunk_key(result)
            for result in relevant_results
        }

        expected_chunks = {
            tuple(chunk)
            for chunk in expected_chunks
        }

        relevant_found = len(
            retrieved_chunks.intersection(expected_chunks)
        )

        predicted_answerable = bool(relevant_results)

        if answerable:
            precision = (
                relevant_found / len(retrieved_chunks)
                if retrieved_chunks else 0
            )

            recall = (
                relevant_found / len(expected_chunks)
                if expected_chunks else 0
            )

            total_precision += precision
            total_recall += recall
            answerable_count += 1

        if predicted_answerable == answerable:
            correct_rejections += 1

    return {
        "precision": total_precision / answerable_count,
        "recall": total_recall / answerable_count,
        "rejection_accuracy": (
            correct_rejections / total_questions
        )
    }


def main():
    methods = ["semantic", "bm25", "hybrid"]

    print("\n========== RETRIEVAL METHOD COMPARISON ==========")

    print(f"Evaluation questions: {len(EVALUATION_DATASET)}")
    print("Answerable questions: 23")
    print("Unanswerable questions: 3")
    print(f"Maximum semantic distance: {MAX_DISTANCE}")

    for k in [3, 5]:

        print(f"\n\nTOP_K = {k}")
        print("=" * 65)

        print(
            f"{'Method':<15}"
            f"{'Precision':<15}"
            f"{'Recall':<15}"
            f"{'Rejection Accuracy':<20}"
        )

        print("-" * 65)

        for method in methods:
            results = evaluate_method(method, k)

            print(
                f"{method.capitalize():<15}"
                f"{results['precision']:<15.2%}"
                f"{results['recall']:<15.2%}"
                f"{results['rejection_accuracy']:<20.2%}"
            )

    print("\nComparison completed.")


if __name__ == "__main__":
    main()
