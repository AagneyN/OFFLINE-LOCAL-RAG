"""Quote-and-verify generation and neighbour expansion."""

import json

import pytest

from backend import generator, rag_pipeline, vector_store


CONTEXT = (
    "[1] A database is an organized collection of data that can be stored, "
    "accessed, managed, and updated electronically.\n\n"
    "[2] PostgreSQL: An open-source relational database known for advanced "
    "SQL features and extensibility."
)


class FakeClient:
    """Returns prepared replies (one per call) and remembers the calls."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        reply = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        content = reply if isinstance(reply, str) else json.dumps(reply)

        return {"message": {"content": content}}


def ask(monkeypatch, reply, *later_replies, question="What is a database?"):
    client = FakeClient(reply, *later_replies)
    monkeypatch.setattr(generator, "_client", client)

    return generator.generate_answer(question, CONTEXT), client


# ---------------------------------------------------- generate_answer

def test_answer_is_returned_when_its_quote_is_in_the_context(monkeypatch):
    answer, _ = ask(monkeypatch, {
        "quotes": ["A database is an organized collection of data that can be stored, accessed, managed, and updated electronically."],
        "answer": "A database is an organized collection of data [1]."
    })

    assert answer == "A database is an organized collection of data [1]."


def test_no_quotes_means_the_model_declined(monkeypatch):
    answer, _ = ask(monkeypatch, {"quotes": [], "answer": ""})

    assert answer == generator.REFUSAL_MESSAGE


def test_an_invented_quote_is_not_accepted(monkeypatch):
    """The model made up install steps and 'quoted' them."""

    answer, _ = ask(
        monkeypatch,
        {
            "quotes": ["Download the installer from the official website and run it."],
            "answer": "Download the installer and run it."
        },
        question="How do I install PostgreSQL on Windows?",
    )

    assert answer == generator.REFUSAL_MESSAGE


def test_an_answer_without_a_supported_quote_is_not_accepted(monkeypatch):
    answer, _ = ask(monkeypatch, {"quotes": ["too short"], "answer": "Something."})

    assert answer == generator.REFUSAL_MESSAGE


def test_blank_answer_is_written_from_the_verified_quotes_only(monkeypatch):
    quote = "A database is an organized collection of data that can be stored, accessed, managed, and updated electronically."

    answer, client = ask(
        monkeypatch,
        {"quotes": [quote, "Something the passages never said at all."], "answer": "  "},
        "A database is an organized collection of data.",
    )

    assert answer == "A database is an organized collection of data."
    assert len(client.calls) == 2

    second = client.calls[1]["messages"][1]["content"]
    assert quote in second
    assert "never said" not in second          # unverified quotes are dropped
    assert "PostgreSQL" not in second          # and the rest of the context is not sent
    assert second.endswith("Question:\nWhat is a database?")


def test_blank_answer_stays_declined_if_the_quotes_do_not_answer(monkeypatch):
    answer, _ = ask(
        monkeypatch,
        {"quotes": ["A database is an organized collection of data that can be stored"], "answer": ""},
        generator.REFUSAL_MESSAGE,
    )

    assert answer == generator.REFUSAL_MESSAGE


def test_blank_answer_and_blank_rewrite_is_declined(monkeypatch):
    answer, _ = ask(
        monkeypatch,
        {"quotes": ["A database is an organized collection of data that can be stored"], "answer": ""},
        "   ",
    )

    assert answer == generator.REFUSAL_MESSAGE


def test_no_second_call_when_the_first_answer_is_usable(monkeypatch):
    _, client = ask(monkeypatch, {
        "quotes": ["A database is an organized collection of data that can be stored"],
        "answer": "A database organizes data."
    })

    assert len(client.calls) == 1


def test_one_good_quote_among_bad_ones_is_enough(monkeypatch):
    answer, _ = ask(monkeypatch, {
        "quotes": [
            "Something the passages never said at all.",
            "PostgreSQL: An open-source relational database known for advanced SQL features and extensibility.",
        ],
        "answer": "PostgreSQL is an open-source relational database."
    })

    assert answer == "PostgreSQL is an open-source relational database."


def test_unstructured_reply_is_returned_unchanged(monkeypatch):
    answer, _ = ask(monkeypatch, "A plain text reply.")

    assert answer == "A plain text reply."


def test_json_of_the_wrong_shape_is_returned_unchanged(monkeypatch):
    answer, _ = ask(monkeypatch, "[1, 2, 3]")

    assert answer == "[1, 2, 3]"


def test_request_asks_for_structured_repeatable_output(monkeypatch):
    _, client = ask(monkeypatch, {"quotes": [], "answer": ""})

    call = client.calls[0]

    assert call["format"] == generator.RESPONSE_SCHEMA
    assert call["options"]["temperature"] == 0
    assert call["options"]["seed"] == generator.GENERATION_OPTIONS["seed"]
    assert "How do I" not in call["messages"][1]["content"]
    assert call["messages"][1]["content"].startswith("Context:\n[1] A database")
    assert call["messages"][1]["content"].endswith("Question:\nWhat is a database?")


# ------------------------------------------------ quote_is_supported

@pytest.mark.parametrize(
    "quote, expected",
    [
        ("a database is an organized collection of data", True),
        ("  A DATABASE   is an organized collection of data. ", True),
        # one word dropped while copying
        ("A database is an collection of data that can be stored, accessed, managed, and updated electronically", True),
        ("A database is a type of software that runs on servers", False),
        # scattered short coincidences with the context must not add up
        ("Download the installer from the official website and run it", False),
        ("short", False),
        ("", False),
    ],
)
def test_quote_is_supported(quote, expected):
    assert generator.quote_is_supported(quote, CONTEXT) is expected


# ------------------------------------------------ get_adjacent_chunks

def result(document_id, chunk_index, text="text"):
    return {
        "text": text,
        "metadata": {"document_id": document_id, "chunk_index": chunk_index},
        "distance": 0.4,
        "rrf_score": 0.03,
    }


@pytest.fixture
def stored(monkeypatch):
    """A tiny fake database of chunks keyed by (document, chunk index)."""

    chunks = {
        ("a.txt", 1): "A one", ("a.txt", 2): "A two", ("a.txt", 3): "A three",
        ("b.txt", 5): "B five", ("b.txt", 6): "B six",
    }
    lookups = []

    def get_chunk(document_id, chunk_index):
        lookups.append((document_id, chunk_index))
        text = chunks.get((document_id, chunk_index))

        return {"text": text, "metadata": {}} if text else None

    monkeypatch.setattr(rag_pipeline, "get_chunk", get_chunk)

    return lookups


def test_following_chunk_is_added_for_the_best_results(stored):
    adjacent = rag_pipeline.get_adjacent_chunks(
        [result("a.txt", 1), result("b.txt", 5), result("a.txt", 3)]
    )

    assert [(c["document_id"], c["chunk_index"], c["text"]) for c in adjacent] == [
        ("a.txt", 2, "A two"),
        ("b.txt", 6, "B six"),
    ]


def test_only_the_top_results_get_a_following_chunk(stored):
    rag_pipeline.get_adjacent_chunks(
        [result("a.txt", 1), result("b.txt", 5), result("a.txt", 3)], count=1
    )

    assert stored == [("a.txt", 2)]


def test_chunks_that_were_already_retrieved_are_not_repeated(stored):
    adjacent = rag_pipeline.get_adjacent_chunks(
        [result("a.txt", 1), result("a.txt", 2)]
    )

    # Chunk 2 is already a result, so it is not fetched or repeated; only
    # the follower of chunk 2 is new.
    assert [c["chunk_index"] for c in adjacent] == [3]
    assert ("a.txt", 2) not in stored


def test_the_last_chunk_of_a_document_has_no_follower(stored):
    adjacent = rag_pipeline.get_adjacent_chunks([result("a.txt", 3)])

    assert adjacent == []


def test_results_without_ids_are_skipped(stored):
    adjacent = rag_pipeline.get_adjacent_chunks(
        [{"text": "x", "metadata": {"source": "x.txt"}, "distance": 0.1, "rrf_score": 0.1}]
    )

    assert adjacent == [] and stored == []


# ------------------------------------------------------ ask_question

def test_adjacent_passages_reach_the_model_but_not_the_sources(monkeypatch, stored):
    seen = {}

    monkeypatch.setattr(
        rag_pipeline, "retrieve_hybrid",
        lambda question, top_k: [result("a.txt", 1, "A one")],
    )

    def fake_generate(question, context):
        seen["context"] = context
        return "An answer."

    monkeypatch.setattr(rag_pipeline, "generate_answer", fake_generate)

    reply = rag_pipeline.ask_question("What is A?")

    assert seen["context"] == "[1] A one\n\n[2] A two"
    assert reply["retrieved_count"] == 1
    assert [s["chunk_index"] for s in reply["sources"]] == [1]
    assert [(a["document_id"], a["chunk_index"]) for a in reply["adjacent"]] == [("a.txt", 2)]


def test_pipeline_retrieves_the_configured_number_of_chunks(monkeypatch):
    requested = {}

    def fake_retrieve(question, top_k):
        requested["top_k"] = top_k
        return []

    monkeypatch.setattr(rag_pipeline, "retrieve_hybrid", fake_retrieve)

    rag_pipeline.ask_question("anything")

    assert requested["top_k"] == rag_pipeline.TOP_K == 5


def test_reply_says_when_the_model_declined(monkeypatch):
    monkeypatch.setattr(
        rag_pipeline, "retrieve_hybrid",
        lambda question, top_k: [result("a.txt", 1)],
    )
    monkeypatch.setattr(
        rag_pipeline, "generate_answer",
        lambda question, context: generator.REFUSAL_MESSAGE,
    )

    reply = rag_pipeline.ask_question("How do I install it?")

    assert reply["declined"] is True
    # The passages that were checked are still shown to the user.
    assert len(reply["sources"]) == 1


def test_reply_is_not_marked_declined_for_a_normal_answer(monkeypatch):
    monkeypatch.setattr(
        rag_pipeline, "retrieve_hybrid",
        lambda question, top_k: [result("a.txt", 1)],
    )
    monkeypatch.setattr(
        rag_pipeline, "generate_answer", lambda question, context: "An answer."
    )

    assert rag_pipeline.ask_question("What is A?")["declined"] is False


# ------------------------------------------------------- get_chunk

def test_get_chunk_looks_up_one_chunk_of_one_document(monkeypatch):
    seen = {}

    class FakeCollection:
        def get(self, where, include):
            seen["where"] = where
            return {"ids": ["x"], "documents": ["chunk text"], "metadatas": [{"chunk_index": 4}]}

    monkeypatch.setattr(vector_store, "get_collection", FakeCollection)

    chunk = vector_store.get_chunk("a.txt", 4)

    assert chunk == {"text": "chunk text", "metadata": {"chunk_index": 4}}
    assert seen["where"] == {
        "$and": [{"document_id": "a.txt"}, {"chunk_index": 4}]
    }


def test_get_chunk_returns_none_when_missing(monkeypatch):
    class FakeCollection:
        def get(self, where, include):
            return {"ids": [], "documents": [], "metadatas": []}

    monkeypatch.setattr(vector_store, "get_collection", FakeCollection)

    assert vector_store.get_chunk("a.txt", 99) is None
