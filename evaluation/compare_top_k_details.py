from pathlib import Path

from backend.config import MAX_DISTANCE
from backend.retriever import retrieve_documents
from evaluation.evaluation_dataset import EVALUATION_DATASET


def inspect_question(test, k):
    results = retrieve_documents(test["question"], top_k=k)

    relevant_results = [
        item for item in results
        if item["distance"] <= MAX_DISTANCE
    ]

    expected = set(test["expected_chunks"])

    retrieved = {
        (
            Path(item["metadata"]["source"]).name,
            item["metadata"]["chunk_index"]
        )
        for item in relevant_results
    }

    found = retrieved.intersection(expected)
    missing = expected - retrieved
    irrelevant = retrieved - expected

    return found, missing, irrelevant


def main():
    print("\n========== TOP_K QUESTION ANALYSIS ==========")

    for test in EVALUATION_DATASET:
        if not test["answerable"]:
            continue

        found_3, missing_3, irrelevant_3 = inspect_question(test, 3)
        found_5, missing_5, irrelevant_5 = inspect_question(test, 5)

        if (
            found_3 != found_5
            or missing_3 != missing_5
            or irrelevant_3 != irrelevant_5
        ):
            print("\nQuestion:", test["question"])
            print("Expected chunks:", sorted(test["expected_chunks"]))

            print("\nTOP_K = 3")
            print("Relevant found:", sorted(found_3))
            print("Missing:", sorted(missing_3))
            print("Irrelevant:", sorted(irrelevant_3))

            print("\nTOP_K = 5")
            print("Relevant found:", sorted(found_5))
            print("Missing:", sorted(missing_5))
            print("Irrelevant:", sorted(irrelevant_5))

            print("-" * 70)


if __name__ == "__main__":
    main()
