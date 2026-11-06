
from backend.config import MAX_DISTANCE
from backend.retriever import retrieve_documents
from backend.hybrid_retriever import retrieve_hybrid

from evaluation.evaluation_dataset import EVALUATION_DATASET


TOP_K = 3


def get_chunk_key(result):
    metadata = result["metadata"]

    return (
        metadata["document_id"],
        metadata["chunk_index"]
    )


def evaluate_question(test):
    question = test["question"]
    expected_chunks = {
        tuple(chunk)
        for chunk in test["expected_chunks"]
    }
    answerable = test["answerable"]

    semantic_results = retrieve_documents(
        question,
        top_k=TOP_K
    )

    hybrid_results = retrieve_hybrid(
        question,
        top_k=TOP_K
    )

    semantic_relevant = [
        result for result in semantic_results
        if result["distance"] <= MAX_DISTANCE
    ]

    hybrid_relevant = [
        result for result in hybrid_results
        if result["distance"] is not None
        and result["distance"] <= MAX_DISTANCE
    ]

    semantic_chunks = {
        get_chunk_key(result)
        for result in semantic_relevant
    }

    hybrid_chunks = {
        get_chunk_key(result)
        for result in hybrid_relevant
    }

    semantic_found = semantic_chunks.intersection(
        expected_chunks
    )

    hybrid_found = hybrid_chunks.intersection(
        expected_chunks
    )

    return {
        "question": question,
        "answerable": answerable,
        "expected": expected_chunks,
        "semantic": semantic_chunks,
        "hybrid": hybrid_chunks,
        "semantic_found": semantic_found,
        "hybrid_found": hybrid_found,
        "semantic_accepted": bool(semantic_relevant),
        "hybrid_accepted": bool(hybrid_relevant)
    }


def main():
    improved = []
    regressed = []
    unchanged = []
    false_positives = []
    false_negatives = []

    print("\n========== HYBRID RETRIEVAL ERROR ANALYSIS ==========")
    print(f"TOP_K: {TOP_K}")
    print(f"Maximum semantic distance: {MAX_DISTANCE}")

    for test in EVALUATION_DATASET:
        result = evaluate_question(test)

        if not result["answerable"]:
            if result["hybrid_accepted"]:
                false_positives.append(result)

            if not result["hybrid_accepted"]:
                false_negatives.append(result)

            continue

        semantic_found = result["semantic_found"]
        hybrid_found = result["hybrid_found"]

        if len(hybrid_found) > len(semantic_found):
            improved.append(result)

        elif len(hybrid_found) < len(semantic_found):
            regressed.append(result)

        else:
            unchanged.append(result)

    def display_group(title, results):
        print(f"\n\n{title} ({len(results)})")
        print("=" * 75)

        for result in results:
            print(f"\nQuestion: {result['question']}")

            print(
                f"Semantic: {len(result['semantic_found'])}/"
                f"{len(result['expected'])} expected chunks found"
            )

            print(
                f"Hybrid:   {len(result['hybrid_found'])}/"
                f"{len(result['expected'])} expected chunks found"
            )

            print(
                "Expected chunks:",
                sorted(result["expected"])
            )

            print(
                "Semantic retrieved:",
                sorted(result["semantic"])
            )

            print(
                "Hybrid retrieved:",
                sorted(result["hybrid"])
            )

    display_group("HYBRID IMPROVEMENTS", improved)
    display_group("HYBRID REGRESSIONS", regressed)
    display_group("UNCHANGED RESULTS", unchanged)

    display_group(
        "FALSE POSITIVES: UNANSWERABLE QUESTIONS ACCEPTED",
        false_positives
    )

    display_group(
        "CORRECTLY REJECTED UNANSWERABLE QUESTIONS",
        false_negatives
    )

    print("\n\n========== SUMMARY ==========")
    print(f"Hybrid improvements: {len(improved)}")
    print(f"Hybrid regressions: {len(regressed)}")
    print(f"Unchanged: {len(unchanged)}")
    print(f"Unanswerable false positives: {len(false_positives)}")
    print(f"Unanswerable correctly rejected: {len(false_negatives)}")


if __name__ == "__main__":
    main()
