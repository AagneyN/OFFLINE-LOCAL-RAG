from pathlib import Path

from backend.config import MAX_DISTANCE
from backend.retriever import retrieve_documents
from evaluation.evaluation_dataset import EVALUATION_DATASET


def evaluate_top_k(k):
    total_precision = 0
    total_recall = 0
    answerable_count = 0

    correct_rejections = 0
    total_questions = len(EVALUATION_DATASET)

    for test in EVALUATION_DATASET:
        question = test["question"]
        expected_chunks = set(test["expected_chunks"])
        answerable = test["answerable"]

        results = retrieve_documents(question, top_k=k)

        relevant_results = [
            item for item in results
            if item["distance"] <= MAX_DISTANCE
        ]

        retrieved_chunks = {
            (
                Path(item["metadata"]["source"]).name,
                item["metadata"]["chunk_index"]
            )
            for item in relevant_results
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
        "rejection_accuracy": correct_rejections / total_questions
    }


def main():
    print("\n========== TOP_K COMPARISON ==========")

    for k in [3, 5]:
        results = evaluate_top_k(k)

        print(f"\nTOP_K = {k}")
        print("-" * 40)
        print(f"Precision:          {results['precision']:.2%}")
        print(f"Recall:             {results['recall']:.2%}")
        print(
            f"Rejection accuracy: "
            f"{results['rejection_accuracy']:.2%}"
        )

    print("\nComparison completed.")


if __name__ == "__main__":
    main()
