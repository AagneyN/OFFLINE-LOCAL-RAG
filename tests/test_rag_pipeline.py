
from backend import rag_pipeline


def make_hybrid_result(
    text,
    source,
    chunk_index,
    distance,
    rrf_score=0.03
):
    return {
        "text": text,
        "metadata": {
            "source": source,
            "document_id": source,
            "chunk_index": chunk_index
        },
        "distance": distance,
        "rrf_score": rrf_score
    }


def test_empty_question():
    result = rag_pipeline.ask_question("")

    assert result["answer"] == "Please enter a question."
    assert result["sources"] == []
    assert result["retrieved_count"] == 0


def test_no_relevant_results(monkeypatch):
    monkeypatch.setattr(
        rag_pipeline,
        "retrieve_hybrid",
        lambda question, top_k: [
            make_hybrid_result(
                "Unrelated content",
                "sample.txt",
                0,
                0.95
            )
        ]
    )

    result = rag_pipeline.ask_question("What is SQL?")

    assert "not find sufficient relevant information" in result["answer"]
    assert result["sources"] == []
    assert result["retrieved_count"] == 0


def test_pipeline_generates_answer(monkeypatch):
    monkeypatch.setattr(
        rag_pipeline,
        "retrieve_hybrid",
        lambda question, top_k: [
            make_hybrid_result(
                "A database stores and manages data.",
                "databases_sql.txt",
                0,
                0.2
            )
        ]
    )

    monkeypatch.setattr(
        rag_pipeline,
        "generate_answer",
        lambda question, context: (
            "A database stores and manages data."
        )
    )

    result = rag_pipeline.ask_question("What is a database?")

    assert result["answer"] == "A database stores and manages data."
    assert result["retrieved_count"] == 1
    assert len(result["sources"]) == 1
    assert result["sources"][0]["source"] == "databases_sql.txt"
    assert result["sources"][0]["rrf_score"] == 0.03


def test_pipeline_excludes_distant_results(monkeypatch):
    monkeypatch.setattr(
        rag_pipeline,
        "retrieve_hybrid",
        lambda question, top_k: [
            make_hybrid_result(
                "Relevant content",
                "sample.txt",
                0,
                0.2
            ),
            make_hybrid_result(
                "Irrelevant content",
                "other.txt",
                1,
                0.95
            )
        ]
    )

    monkeypatch.setattr(
        rag_pipeline,
        "generate_answer",
        lambda question, context: (
            "Answer based on relevant content."
        )
    )

    result = rag_pipeline.ask_question("What is this about?")

    assert result["retrieved_count"] == 1
    assert len(result["sources"]) == 1
    assert result["sources"][0]["source"] == "sample.txt"


def test_pipeline_ignores_results_without_distance(monkeypatch):
    result_without_distance = {
        "text": "Content without semantic distance",
        "metadata": {
            "source": "sample.txt",
            "chunk_index": 0
        },
        "rrf_score": 0.03,
        "distance": None
    }

    monkeypatch.setattr(
        rag_pipeline,
        "retrieve_hybrid",
        lambda question, top_k: [result_without_distance]
    )

    result = rag_pipeline.ask_question("What is this about?")

    assert result["sources"] == []
    assert result["retrieved_count"] == 0


def test_sources_include_chunk_text_and_scores(monkeypatch):
    hybrid_result = make_hybrid_result(
        "A database stores and manages data.",
        "databases_sql.txt",
        3,
        0.2
    )
    hybrid_result["bm25_score"] = 4.2

    monkeypatch.setattr(
        rag_pipeline,
        "retrieve_hybrid",
        lambda question, top_k: [hybrid_result]
    )

    monkeypatch.setattr(
        rag_pipeline,
        "generate_answer",
        lambda question, context: "A database stores data."
    )

    source = rag_pipeline.ask_question("What is a database?")["sources"][0]

    assert source["document_id"] == "databases_sql.txt"
    assert source["chunk_index"] == 3
    assert source["text"] == "A database stores and manages data."
    assert source["bm25_score"] == 4.2
    assert source["distance"] == 0.2


def test_generation_is_skipped_when_nothing_is_relevant(monkeypatch):
    monkeypatch.setattr(
        rag_pipeline,
        "retrieve_hybrid",
        lambda question, top_k: [
            make_hybrid_result("Unrelated", "sample.txt", 0, 0.95)
        ]
    )

    def fail(question, context):
        raise AssertionError("The LLM must not be called without context")

    monkeypatch.setattr(rag_pipeline, "generate_answer", fail)

    result = rag_pipeline.ask_question("What is the capital of France?")

    assert result["retrieved_count"] == 0
