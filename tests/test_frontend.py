"""UI tests using Streamlit's headless AppTest.

All backend calls are stubbed and the upload folders point at tmp_path, so
these tests never touch the real vector database or the docs/ folder.
"""

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import ollama
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from backend import (
    document_manager,
    generator,
    indexer,
    rag_pipeline,
    vector_store,
)


# AppTest runs these functions in a fresh script, so each one imports what
# it needs itself.

def dashboard_page():
    from frontend.app import render_dashboard
    render_dashboard()


def chat_page():
    from frontend.app import render_chat
    render_chat()


def documents_page():
    from frontend.app import render_documents
    render_documents()


def settings_page():
    from frontend.app import render_settings
    render_settings()


def startup_error_page():
    from frontend.app import render_startup_error
    render_startup_error(OSError("model not cached"))


HEALTHY_OLLAMA = {
    "reachable": True,
    "model_installed": True,
    "installed_models": ["qwen2.5:3b-instruct"],
    "processor": "GPU",
    "error": None,
}

SOURCE = {
    "source": "C:\\docs\\machine_learning.txt",
    "document_id": "machine_learning.txt",
    "chunk_index": 4,
    "distance": 0.503,
    "rrf_score": 0.0328,
    "bm25_score": 5.75,
    "text": "Overfitting occurs when a model learns noise.",
}


@pytest.fixture
def stub(monkeypatch, tmp_path):
    # The page caches Ollama's status for a few seconds; start each test clean.
    st.cache_data.clear()

    state = SimpleNamespace(
        chunks=38,
        ollama=dict(HEALTHY_OLLAMA),
        questions=[],
        result={
            "answer": "Overfitting is when a model memorises noise.",
            "sources": [SOURCE],
            "retrieved_count": 1,
        },
        error=None,
        index_calls=[],
        index_report={
            "documents_found": 1,
            "documents_indexed": 1,
            "documents_unchanged": 0,
            "documents_removed": 0,
            "documents_failed": 0,
            "failed": [],
            "new_chunks": 2,
            "total_chunks": 40,
        },
        index_error=None,
    )

    def ask_question(question):
        state.questions.append(question)

        if state.error:
            raise state.error

        return state.result

    def index_documents(progress_callback=None, continue_on_error=False):
        state.index_calls.append(continue_on_error)

        if progress_callback:
            progress_callback(1, 1, "uploads/x.txt")

        if state.index_error:
            raise state.index_error

        return state.index_report

    docs = tmp_path / "docs"
    docs.mkdir()

    monkeypatch.setattr(vector_store, "get_collection_count", lambda: state.chunks)
    monkeypatch.setattr(generator, "get_ollama_status", lambda: state.ollama)
    monkeypatch.setattr(rag_pipeline, "ask_question", ask_question)
    monkeypatch.setattr(indexer, "index_documents", index_documents)
    monkeypatch.setattr(document_manager, "get_indexed_documents", lambda: {})
    monkeypatch.setattr(document_manager, "DOCS_DIR", docs)
    monkeypatch.setattr(document_manager, "UPLOAD_DIR", docs / "uploads")

    state.docs = docs
    return state


def run(page):
    at = AppTest.from_function(page, default_timeout=120).run()
    assert not at.exception
    return at


def button(at, label):
    return next(b for b in at.button if b.label == label)


def texts(elements):
    return " ".join(str(element.value) for element in elements)


# ---------------------------------------------------------------- chat

def test_chat_answers_and_shows_sources(stub):
    at = run(chat_page)

    at.chat_input[0].set_value("What is overfitting?").run()

    assert not at.exception
    assert stub.questions == ["What is overfitting?"]

    roles = [message.name for message in at.chat_message]
    assert roles == ["user", "assistant"]

    answer = at.chat_message[1]
    assert "memorises noise" in answer.markdown[0].value
    assert any(e.label == "Sources used (1)" for e in at.expander)

    sources = " ".join(m.value for m in at.markdown)
    assert "machine_learning.txt" in sources
    assert "chunk 4" in sources

    scores = texts(at.caption)
    assert "Cosine distance 0.503" in scores
    assert "RRF score 0.0328" in scores
    assert "BM25 5.75" in scores

    assert at.code[0].value == SOURCE["text"]


def test_chat_rejection_explains_that_llm_was_not_used(stub):
    stub.result = {
        "answer": "I could not find sufficient relevant information "
                  "in the knowledge base.",
        "sources": [],
        "retrieved_count": 0,
    }

    at = run(chat_page)
    at.chat_input[0].set_value("What is the capital of France?").run()

    assert "could not find sufficient" in at.chat_message[1].markdown[0].value
    assert "language model was not asked" in texts(at.caption)
    assert not at.expander


def test_chat_ignores_whitespace_only_question(stub):
    at = run(chat_page)
    at.chat_input[0].set_value("   ").run()

    assert stub.questions == []
    assert "Please enter a question." in texts(at.warning)


def test_chat_is_disabled_when_database_is_empty(stub):
    stub.chunks = 0

    at = run(chat_page)

    assert at.chat_input[0].disabled
    assert "knowledge base is empty" in texts(at.info)


@pytest.mark.parametrize(
    "error, expected",
    [
        (ConnectionError("refused"), "cannot reach Ollama"),
        (ollama.ResponseError("not found", 404), "ollama pull"),
        (RuntimeError("boom"), "Something went wrong"),
    ],
)
def test_chat_shows_friendly_error_and_keeps_working(stub, error, expected):
    stub.error = error

    at = run(chat_page)
    at.chat_input[0].set_value("What is overfitting?").run()

    assert not at.exception
    assert expected.lower() in texts(at.error).lower()
    assert "Traceback" not in texts(at.error)

    # The app is still usable: the next question succeeds.
    stub.error = None
    at.chat_input[0].set_value("Try again").run()

    assert not at.exception
    assert stub.questions == ["What is overfitting?", "Try again"]
    assert "memorises noise" in at.chat_message[-1].markdown[0].value


def test_clear_conversation_empties_history(stub):
    at = run(chat_page)
    at.chat_input[0].set_value("What is overfitting?").run()

    assert len(at.chat_message) == 2
    assert not button(at, "Clear conversation").disabled

    button(at, "Clear conversation").click().run()

    assert len(at.chat_message) == 0
    assert button(at, "Clear conversation").disabled


def test_chat_shows_adjacent_passages_separately(stub):
    stub.result = {
        **stub.result,
        "adjacent": [{
            "document_id": "machine_learning.txt",
            "chunk_index": 5,
            "text": "The passage that continues the match.",
        }],
    }

    at = run(chat_page)
    at.chat_input[0].set_value("What is overfitting?").run()

    shown = " ".join(m.value for m in at.markdown)

    assert "[2] machine_learning.txt" in shown
    assert "chunk 5 (continuation)" in shown
    assert at.code[1].value == "The passage that continues the match."
    assert any(e.label == "Sources used (1)" for e in at.expander)


def test_chat_explains_when_the_model_declined(stub):
    stub.result = {
        "answer": "I don't have enough information in the documents to answer that.",
        "sources": [SOURCE],
        "adjacent": [],
        "declined": True,
        "retrieved_count": 1,
    }

    at = run(chat_page)
    at.chat_input[0].set_value("How do I install it?").run()

    assert any(e.label == "Passages checked (1)" for e in at.expander)
    assert not any(e.label.startswith("Sources used") for e in at.expander)
    assert "declined to answer instead of guessing" in texts(at.caption)
    # The model was asked, so the "was not asked" note must not appear.
    assert "language model was not asked" not in texts(at.caption)


# ----------------------------------------------------------- dashboard

def test_dashboard_healthy(stub):
    at = run(dashboard_page)

    metrics = {m.label: m.value for m in at.metric}
    assert metrics == {"Indexed documents": "0", "Chunks in database": "38"}

    status = " ".join(m.value for m in at.markdown)
    assert "qwen2.5:3b-instruct is installed" in status
    assert "38 chunks" in status


def test_dashboard_reports_ollama_down(stub):
    stub.ollama = {**HEALTHY_OLLAMA, "reachable": False,
                   "model_installed": False, "installed_models": []}

    status = " ".join(m.value for m in run(dashboard_page).markdown)

    assert "not reachable" in status


def test_dashboard_reports_missing_model(stub):
    stub.ollama = {**HEALTHY_OLLAMA, "model_installed": False}

    status = " ".join(m.value for m in run(dashboard_page).markdown)

    assert "ollama pull qwen2.5:3b-instruct" in status


def test_dashboard_guides_user_when_database_is_empty(stub):
    stub.chunks = 0

    at = run(dashboard_page)

    assert "Getting started" in texts(at.info)


# ----------------------------------------------------------- settings

def test_settings_show_configuration(stub):
    at = run(settings_page)

    shown = " ".join(
        table.value.to_string() for table in at.table
    )

    assert "qwen2.5:3b-instruct" in shown
    assert "all-MiniLM-L6-v2" in shown
    assert "Reciprocal Rank Fusion" in shown
    assert "0.75" in shown
    assert "GPU" in shown


def test_settings_when_ollama_down(stub):
    stub.ollama = {**HEALTHY_OLLAMA, "reachable": False}

    at = run(settings_page)

    assert "not reachable" in texts(at.error)


def test_startup_error_explains_offline_model_setup():
    at = run(startup_error_page)

    assert "could not start" in texts(at.error)
    assert "SentenceTransformer" in at.code[0].value


# ---------------------------------------------------------- documents

def test_documents_lists_library(stub):
    (stub.docs / "sample.txt").write_text("hello", encoding="utf-8")

    at = run(documents_page)

    rows = at.dataframe[0].value
    assert list(rows["Document"]) == ["sample.txt"]
    assert list(rows["Status"]) == ["Not indexed"]
    assert "1 document not indexed yet" in texts(at.warning)


def test_upload_valid_file_then_save_and_index(stub):
    at = run(documents_page)

    assert button(at, "Save and index").disabled

    at.file_uploader[0].upload("notes.txt", b"A database stores data.").run()

    assert not at.exception
    assert "Ready: 1 chunk" in texts(at.success)

    button(at, "Save and index").click().run()

    assert not at.exception
    assert (stub.docs / "uploads" / "notes.txt").read_bytes() == (
        b"A database stores data."
    )
    # The UI asks the backend to isolate failures per document.
    assert stub.index_calls == [True]
    assert "Indexed 1 document (2 new chunks)" in texts(at.success)


@pytest.mark.parametrize(
    "name, data, expected",
    [
        ("broken.pdf", b"not a pdf", "could not be read"),
        ("empty.txt", b"", "The file is empty."),
        ("blank.txt", b"   \n  ", "no text"),
        ("latin.txt", "x".encode("utf-16"), "not valid UTF-8"),
    ],
)
def test_upload_rejects_unusable_files(stub, name, data, expected):
    at = run(documents_page)
    at.file_uploader[0].upload(name, data).run()

    assert not at.exception
    assert expected in texts(at.error)
    assert button(at, "Save and index").disabled


def test_upload_identical_file_is_not_saved_again(stub):
    uploads = stub.docs / "uploads"
    uploads.mkdir()
    (uploads / "notes.txt").write_bytes(b"Same content.")

    at = run(documents_page)
    at.file_uploader[0].upload("notes.txt", b"Same content.").run()

    assert "already in the library" in texts(at.info)
    assert button(at, "Save and index").disabled


def test_name_conflict_defaults_to_keeping_both(stub):
    uploads = stub.docs / "uploads"
    uploads.mkdir()
    (uploads / "notes.txt").write_bytes(b"Original.")

    at = run(documents_page)
    at.file_uploader[0].upload("notes.txt", b"Changed.").run()

    assert at.radio[0].value == "Keep both (save as a copy)"

    button(at, "Save and index").click().run()

    assert (uploads / "notes.txt").read_bytes() == b"Original."
    assert (uploads / "notes (2).txt").read_bytes() == b"Changed."


def test_name_conflict_can_replace_existing_file(stub):
    uploads = stub.docs / "uploads"
    uploads.mkdir()
    (uploads / "notes.txt").write_bytes(b"Original.")

    at = run(documents_page)
    at.file_uploader[0].upload("notes.txt", b"Changed.").run()
    at.radio[0].set_value("Replace the existing file").run()
    button(at, "Save and index").click().run()

    assert (uploads / "notes.txt").read_bytes() == b"Changed."
    assert [p.name for p in uploads.iterdir()] == ["notes.txt"]


def test_failed_document_is_reported_but_others_succeed(stub):
    stub.index_report = {
        **stub.index_report,
        "documents_failed": 1,
        "failed": [{"document_id": "bad.pdf", "error": "unreadable"}],
    }

    at = run(documents_page)
    at.file_uploader[0].upload("good.txt", b"Some real text.").run()
    button(at, "Save and index").click().run()

    warning = texts(at.warning)
    assert "bad.pdf" in warning
    assert "unreadable" in warning


def test_indexing_failure_shows_friendly_message(stub):
    stub.index_error = ConnectionError("down")

    at = run(documents_page)
    at.file_uploader[0].upload("good.txt", b"Some real text.").run()
    button(at, "Save and index").click().run()

    assert not at.exception
    assert "Indexing could not run" in texts(at.error)


def test_rescan_runs_indexer_without_uploads(stub):
    at = run(documents_page)

    button(at, "Re-scan and index library").click().run()

    assert not at.exception
    assert stub.index_calls == [True]
    assert "Indexed 1 document" in texts(at.success)


# -------------------------------------------------------- error helper

@pytest.mark.parametrize(
    "error, expected",
    [
        (ConnectionError("refused"), "cannot reach Ollama"),
        (ollama.ResponseError("not found", 404), "ollama pull"),
        (ollama.ResponseError("server error", 500), "Something went wrong"),
        (ValueError("bad value"), "ValueError: bad value"),
    ],
)
def test_explain_error(error, expected):
    from frontend.app import explain_error

    assert expected.lower() in explain_error(error).lower()


def test_running_app_file_directly_explains_how_to_start_it():
    """`python frontend/app.py` (the editor's Run button) must not run the app."""

    root = Path(__file__).resolve().parent.parent

    result = subprocess.run(
        [sys.executable, "frontend/app.py"],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert result.returncode != 0
    assert "python -m streamlit run frontend/app.py" in result.stderr
    assert "ScriptRunContext" not in result.stderr
