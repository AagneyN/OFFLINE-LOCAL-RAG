"""Checks on the answer-quality dataset and scorer (no LLM or database needed)."""

import re

from backend.config import DOCS_DIR
from evaluation import answer_quality
from evaluation.answer_quality_dataset import DEV_SET, FINAL_SET, HELDOUT_SET

ALL_ITEMS = DEV_SET + HELDOUT_SET + FINAL_SET


def corpus() -> str:
    """The sample documents, normalised the same way as retrieved chunks."""

    sample_documents = [
        "rag_basics.txt",
        "machine_learning.txt",
        "deep_learning.txt",
        "llm_fundamentals.txt",
        "databases_sql.txt",
    ]

    return answer_quality.normalize(
        " ".join(
            (DOCS_DIR / name).read_text(encoding="utf-8-sig")
            for name in sample_documents
        )
    )


def test_every_evidence_phrase_exists_in_the_documents():
    text = corpus()

    missing = [
        (item["question"], phrase)
        for item in ALL_ITEMS
        for phrase in item["evidence"]
        if phrase not in text
    ]

    assert missing == []


def test_answer_patterns_are_valid_regular_expressions():
    for item in ALL_ITEMS:
        for pattern in item["answer"]:
            re.compile(pattern)


def test_questions_are_unique_across_sets():
    questions = [item["question"] for item in ALL_ITEMS]

    assert len(questions) == len(set(questions))


def test_answerable_questions_have_checks_and_others_have_none():
    for item in ALL_ITEMS:
        if item["answerable"]:
            assert item["evidence"] and item["answer"], item["question"]
        else:
            assert not item["evidence"] and not item["answer"], item["question"]


def test_score_detects_hit_and_correct_answer():
    item = {
        "question": "q",
        "answerable": True,
        "evidence": ["numerical vectors"],
        "answer": [r"vector"],
    }

    good = answer_quality.score(
        item,
        {"answer": "Text becomes a Vector.",
         "sources": [{"text": "Embeddings represent text as  Numerical\nvectors."}]},
    )
    bad = answer_quality.score(
        item,
        {"answer": "I do not know.", "sources": [{"text": "unrelated"}]},
    )

    assert good["context_hit"] and good["answer_ok"]
    assert not bad["context_hit"] and not bad["answer_ok"]


def test_score_counts_early_rejection_and_model_refusal():
    item = {"question": "q", "answerable": False, "evidence": [], "answer": []}

    early = answer_quality.score(
        item, {"answer": "I could not find sufficient information.", "sources": []}
    )
    declined = answer_quality.score(
        item,
        {"answer": "The context does not mention this.",
         "sources": [{"text": "something"}]},
    )
    invented = answer_quality.score(
        item,
        {"answer": "It was built in 1999 by Alice.",
         "sources": [{"text": "something"}]},
    )

    assert early["refused"] and early["rejected_before_llm"]
    assert declined["refused"] and not declined["rejected_before_llm"]
    assert not invented["refused"]


def test_summarize_rates():
    rows = [
        {"answerable": True, "context_hit": True, "answer_ok": True, "sources": 3},
        {"answerable": True, "context_hit": True, "answer_ok": False, "sources": 1},
        {"answerable": False, "refused": True, "rejected_before_llm": True, "sources": 0},
    ]

    summary = answer_quality.summarize(rows)

    assert summary["context_hit"] == 1.0
    assert summary["answer_ok"] == 0.5
    assert summary["refused"] == 1.0
    assert summary["avg_sources"] == 2.0


def test_patterns_contain_no_control_characters():
    """A backslash-b written as a backspace silently makes a pattern unmatchable."""

    for item in ALL_ITEMS:
        for pattern in item["answer"]:
            assert not re.search(r"[\x00-\x08\x0b-\x1f]", pattern), item["question"]


def test_join_pattern_accepts_correct_answer_and_rejects_a_garbled_one():
    item = next(i for i in DEV_SET if "INNER JOIN and LEFT JOIN" in i["question"])

    correct = (
        "INNER JOIN returns records that have matching values in both tables, "
        "while LEFT JOIN returns all records from the left table and matching "
        "records from the right table."
    )
    garbled = (
        "INNER JOIN returns all records from the left table and matching "
        "records from the right table. LEFT JOIN returns all records from the "
        "left table and matching records from the right table."
    )

    def passes(answer):
        return all(re.search(p, answer, re.IGNORECASE) for p in item["answer"])

    assert passes(correct)
    assert not passes(garbled)


def test_scorer_recognises_the_pipelines_own_refusal_message():
    """The evaluation must count the sentence the generator really produces."""

    from backend.generator import REFUSAL_MESSAGE

    item = {"question": "q", "answerable": False, "evidence": [], "answer": []}
    row = answer_quality.score(
        item,
        {"answer": REFUSAL_MESSAGE, "sources": [{"text": "a related passage"}]},
    )

    assert row["refused"]
    assert not row["rejected_before_llm"]


def test_context_hit_counts_adjacent_passages_given_to_the_model():
    item = {
        "question": "q",
        "answerable": True,
        "evidence": ["durability ensures", "isolation ensures"],
        "answer": [r"durability"],
    }

    row = answer_quality.score(
        item,
        {
            "answer": "Durability and isolation.",
            "sources": [{"text": "Isolation ensures that transactions do not interfere."}],
            "adjacent": [{"text": "Durability ensures that committed changes remain saved."}],
        },
    )

    assert row["context_hit"]
    assert row["sources"] == 1
