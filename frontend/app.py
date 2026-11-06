"""LocalRAG web interface.

This file only contains the user interface. Every RAG step (loading,
chunking, embedding, indexing, hybrid retrieval, generation) is done by the
modules in backend/.

Run with:  python -m streamlit run frontend/app.py
"""

import logging
import sys
from pathlib import Path
from types import SimpleNamespace

# Make the backend package importable when launched with `streamlit run`.
ROOT = Path(__file__).resolve().parent.parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import ollama
import pandas as pd
import streamlit as st

# `python frontend/app.py` (for example the editor's Run button) executes the
# file without a Streamlit session. Say how to start it properly instead of
# loading the models and printing a page of warnings. This runs before the
# cached functions below are defined, so it is also quiet.
if __name__ == "__main__":
    from streamlit.runtime.scriptrunner import get_script_run_ctx

    if get_script_run_ctx(suppress_warning=True) is None:
        sys.exit(
            "LocalRAG is a Streamlit app and cannot be started with "
            "`python app.py`.\nFrom the project folder run:\n\n"
            "    python -m streamlit run frontend/app.py\n\n"
            "It then opens at http://localhost:8501."
        )

logger = logging.getLogger("localrag")


# ---------------------------------------------------------------------------
# Backend access
# ---------------------------------------------------------------------------

@st.cache_resource(
    show_spinner="Loading the local embedding model and vector database..."
)
def load_backend() -> SimpleNamespace:
    """Import the backend once per server process.

    Importing it loads the embedding model (from the local cache only) and
    opens the persistent ChromaDB, which takes a few seconds.
    """

    from backend import (
        config,
        document_manager,
        embeddings,
        generator,
        hybrid_retriever,
        indexer,
        rag_pipeline,
        vector_store,
    )

    return SimpleNamespace(
        config=config,
        documents=document_manager,
        embeddings=embeddings,
        generator=generator,
        hybrid=hybrid_retriever,
        indexer=indexer,
        pipeline=rag_pipeline,
        store=vector_store,
    )


@st.cache_data(ttl=5, show_spinner=False)
def ollama_status() -> dict:
    """Ollama health, cached briefly.

    Every interaction reruns the page, and while Ollama is stopped a single
    check takes a few seconds on Windows. The cache keeps that to one check
    per rerun instead of one per widget that shows the status.
    """

    return load_backend().generator.get_ollama_status()


def explain_error(error: Exception) -> str:
    """Turn an exception into a message an end user can act on."""

    config = load_backend().config

    if isinstance(error, ConnectionError):
        return (
            f"LocalRAG cannot reach Ollama at {config.OLLAMA_HOST}. "
            "Start the Ollama app (or run `ollama serve` in a terminal) "
            "and try again."
        )

    if isinstance(error, ollama.ResponseError) and error.status_code == 404:
        return (
            f"The language model `{config.LLM_MODEL}` is not installed in "
            f"Ollama. Run `ollama pull {config.LLM_MODEL}` and try again."
        )

    detail = f"{type(error).__name__}: {error}"

    return (
        "Something went wrong while processing your request. "
        f"({detail[:200]})"
    )


def plural(count: int, word: str) -> str:
    return f"{count} {word}{'' if count == 1 else 's'}"


def format_size(kilobytes) -> str:
    if kilobytes is None:
        return "-"

    return (
        f"{kilobytes / 1024:.1f} MB"
        if kilobytes >= 1024
        else f"{kilobytes:.1f} KB"
    )


# ---------------------------------------------------------------------------
# Shared widgets
# ---------------------------------------------------------------------------

def render_sidebar_status():
    backend = load_backend()
    status = ollama_status()

    with st.sidebar:
        if not status["reachable"]:
            st.error("Ollama is not running")
        elif not status["model_installed"]:
            st.warning(f"Model {backend.config.LLM_MODEL} is not installed")
        else:
            st.success("Ollama is ready")

        st.caption(
            "All processing happens on this computer. "
            "No documents or questions are sent to the cloud."
        )


def key_value_table(rows: dict):
    # st.table wraps long values (paths, descriptions); st.dataframe cuts them.
    st.table(
        pd.DataFrame(
            {"Value": [str(value) for value in rows.values()]},
            index=list(rows),
        )
    )


def status_line(level: str, label: str, detail: str):
    icon = {"ok": "✅", "warn": "⚠️", "error": "❌"}[level]
    st.markdown(f"{icon} **{label}** — {detail}")


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

def render_dashboard():
    backend = load_backend()
    config = backend.config

    # Decorative shapes (styled in theme.css); hidden from assistive technology.
    st.markdown(
        '<div class="organic-art" aria-hidden="true"><span></span><span></span><span></span></div>',
        unsafe_allow_html=True,
    )
    st.title("🔎 LocalRAG")
    st.write(
        "Ask questions about your own TXT and PDF documents. Everything "
        "runs locally: documents are split into chunks, embedded on this "
        "computer, stored in ChromaDB and searched with hybrid retrieval "
        "(semantic search + BM25). A local language model then answers "
        "using only the passages it was given."
    )

    documents = backend.documents.list_documents()
    indexed = [row for row in documents if row["status"] == "Indexed"]
    chunk_count = backend.store.get_collection_count()


    # Dashboard metrics: two rows with two columns each.
    document_columns = st.columns(2)

    document_columns[0].metric("Indexed documents", len(indexed))
    document_columns[1].metric("Chunks in database", chunk_count)

    model_columns = st.columns(2)

    with model_columns[0]:
        st.caption("Language model")
        st.markdown(f"**{config.LLM_MODEL}**")

    with model_columns[1]:
        st.caption("Embedding model")
        st.markdown(f"**{config.EMBEDDING_MODEL.split('/')[-1]}**")


    st.subheader("System status")

    status = ollama_status()

    with st.container(border=True):
        st.markdown('<span class="organic-status" aria-hidden="true"></span>', unsafe_allow_html=True)

        if status["reachable"]:
            status_line("ok", "Ollama", f"running at {config.OLLAMA_HOST}")
        else:
            status_line(
                "error",
                "Ollama",
                f"not reachable at {config.OLLAMA_HOST}. Start the Ollama "
                "app or run `ollama serve`.",
            )

        if status["model_installed"]:
            status_line("ok", "Language model", f"{config.LLM_MODEL} is installed")
        elif status["reachable"]:
            status_line(
                "error",
                "Language model",
                f"{config.LLM_MODEL} is not installed. "
                f"Run `ollama pull {config.LLM_MODEL}`.",
            )

        status_line(
            "ok",
            "Embedding model",
            f"{config.EMBEDDING_MODEL} loaded from the local cache "
            f"(device: {backend.embeddings.get_embedding_device()})",
        )

        if chunk_count:
            status_line(
                "ok",
                "Vector database",
                f"{chunk_count} chunks from {len(indexed)} documents "
                f"({config.DB_DIR})",
            )
        else:
            status_line(
                "warn",
                "Vector database",
                "empty. Open **Documents** in the sidebar to upload and "
                "index files.",
            )

    if chunk_count:
        st.info(
            "Open **Chat** in the sidebar to ask questions, or "
            "**Documents** to add more files."
        )
    else:
        st.info(
            "**Getting started:** open **Documents**, upload TXT or PDF "
            "files and press **Save and index**. Then ask questions in "
            "**Chat**."
        )


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------

def clear_conversation():
    st.session_state.messages = []


def render_sources(
    sources: list[dict],
    adjacent: list[dict] | None = None,
    declined: bool = False,
):
    label = "Passages checked" if declined else "Sources used"

    with st.expander(f"{label} ({len(sources)})"):
        if declined:
            st.caption(
                "The model read these passages but they do not answer the "
                "question, so it declined to answer instead of guessing."
            )
        else:
            st.caption(
                "These are the passages the model was given. Lower cosine "
                "distance means a closer semantic match."
            )

        for number, source in enumerate(sources, start=1):
            bm25 = source.get("bm25_score")

            with st.container(border=True):
                st.markdown(
                    f"**[{number}] {source['document_id']}** "
                    f"· chunk {source['chunk_index']}"
                )
                st.caption(
                    f"Cosine distance {source['distance']:.3f} "
                    f"(similarity {1 - source['distance']:.2f}) · "
                    f"RRF score {source['rrf_score']:.4f} · "
                    f"BM25 {'-' if bm25 is None else f'{bm25:.2f}'}"
                )
                st.code(source["text"], language=None, wrap_lines=True)

        if adjacent:
            st.caption(
                "Also given to the model: the passage that follows each of "
                "the best matches, so an answer that continues across a "
                "chunk boundary is not cut off."
            )

            for number, chunk in enumerate(adjacent, start=len(sources) + 1):
                with st.container(border=True):
                    st.markdown(
                        f"**[{number}] {chunk['document_id']}** "
                        f"· chunk {chunk['chunk_index']} (continuation)"
                    )
                    st.code(chunk["text"], language=None, wrap_lines=True)


def render_message(message: dict):
    with st.chat_message(message["role"]):
        if message.get("error"):
            st.error(message["content"])
            return

        st.markdown(message["content"])

        if message.get("sources"):
            render_sources(
                message["sources"],
                message.get("adjacent"),
                message.get("declined", False),
            )
        elif message["role"] == "assistant":
            st.caption(
                "No passage in the knowledge base was relevant enough, so "
                "the language model was not asked."
            )


def answer_question(question: str) -> dict:
    """Run the backend pipeline and wrap the outcome as a chat message."""

    pipeline = load_backend().pipeline

    try:
        with st.spinner("Searching your documents and generating an answer..."):
            result = pipeline.ask_question(question)

    except Exception as error:
        logger.exception("Question failed")
        return {
            "role": "assistant",
            "content": explain_error(error),
            "error": True,
        }

    return {
        "role": "assistant",
        "content": result["answer"],
        "sources": result["sources"],
        "adjacent": result.get("adjacent", []),
        "declined": result.get("declined", False),
    }


def render_chat():
    backend = load_backend()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    header, action = st.columns([5, 1], vertical_alignment="center")
    header.title("💬 Chat")
    action.button(
        "Clear conversation",
        on_click=clear_conversation,
        disabled=not st.session_state.messages,
        width="stretch",
    )

    knowledge_base_empty = backend.store.get_collection_count() == 0

    if knowledge_base_empty:
        st.info(
            "The knowledge base is empty. Open **Documents** in the "
            "sidebar and index some files before asking questions."
        )
    elif not st.session_state.messages:
        st.caption(
            "Answers are based only on your indexed documents. If they "
            "do not contain the answer, LocalRAG says so instead of guessing."
        )

    for message in st.session_state.messages:
        render_message(message)

    question = st.chat_input(
        "Ask a question about your documents...",
        max_chars=1000,
        disabled=knowledge_base_empty,
    )

    if question is None:
        return

    if not question.strip():
        st.warning("Please enter a question.")
        return

    user_message = {"role": "user", "content": question.strip()}
    render_message(user_message)

    with st.chat_message("assistant"):
        assistant_message = answer_question(user_message["content"])

    # Redraw the reply in its final form (answer + sources) and keep both
    # messages in the history.
    st.session_state.messages += [user_message, assistant_message]
    st.rerun()


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False, max_entries=50)
def inspect_upload(name: str, data: bytes) -> dict:
    """Validate an upload once (PDF parsing is slow to repeat on every rerun)."""

    documents = load_backend().documents

    try:
        return {"ok": True, "chunks": documents.validate_document(name, data)}
    except documents.DocumentError as error:
        return {"ok": False, "error": str(error)}


def run_indexing():
    """Index the whole library, showing progress. Returns the report."""

    indexer = load_backend().indexer
    progress = st.progress(0.0, text="Preparing...")

    def update(position: int, total: int, document_id: str):
        progress.progress(
            (position - 1) / total,
            text=f"Indexing {position}/{total}: {document_id}",
        )

    try:
        report = indexer.index_documents(
            progress_callback=update, continue_on_error=True
        )

    except Exception as error:
        logger.exception("Indexing failed")
        progress.empty()
        return {"fatal": explain_error(error)}

    progress.progress(1.0, text="Indexing finished")
    return report


def render_index_report(report: dict):
    if "fatal" in report:
        st.error(f"Indexing could not run. {report['fatal']}")
        return

    docs_folder = str(load_backend().config.DOCS_DIR)

    for failure in report["failed"]:
        # Library errors quote the full file path; "docs" is enough here.
        reason = failure["error"].replace(docs_folder, "docs")

        st.warning(
            f"**{failure['document_id']}** could not be indexed and was "
            f"skipped: {reason}"
        )

    summary = (
        f"Indexed {plural(report['documents_indexed'], 'document')} "
        f"({plural(report['new_chunks'], 'new chunk')}) · "
        f"{report['documents_unchanged']} unchanged · "
        f"{plural(report['total_chunks'], 'chunk')} in the database."
    )

    if report["documents_removed"]:
        summary += (
            f" Removed {plural(report['documents_removed'], 'document')} "
            "whose files no longer exist."
        )

    (st.warning if report["failed"] else st.success)(summary)


def render_upload_section() -> list[tuple]:
    """Show the uploader and return the files that are ready to save."""

    documents = load_backend().documents

    version = st.session_state.setdefault("uploader_version", 0)

    uploads = st.file_uploader(
        "Upload TXT or PDF files",
        type=["txt", "pdf"],
        accept_multiple_files=True,
        key=f"uploader_{version}",
        help="Files are copied to docs/uploads/ and never overwrite the "
        "documents that ship with the project.",
    )

    ready = []

    for upload in uploads:
        data = upload.getvalue()
        info = inspect_upload(upload.name, data)

        with st.container(border=True):
            left, right = st.columns([3, 4], vertical_alignment="center")
            left.markdown(f"**{upload.name}**  \n{format_size(upload.size / 1024)}")

            if not info["ok"]:
                right.error(info["error"])
                continue

            state = documents.classify_upload(upload.name, data)

            if state == "identical":
                right.info("This exact file is already in the library.")

            elif state == "conflict":
                choice = right.radio(
                    "A different file with this name already exists.",
                    ["Keep both (save as a copy)", "Replace the existing file"],
                    key=f"conflict_{version}_{upload.file_id}",
                )
                ready.append((
                    upload,
                    "replace" if choice.startswith("Replace") else "keep_both",
                ))

            else:
                right.success(f"Ready: {plural(info['chunks'], 'chunk')}")
                ready.append((upload, "keep_both"))

    return ready


def save_and_index(ready: list[tuple]):
    documents = load_backend().documents
    errors = []

    for upload, policy in ready:
        try:
            documents.save_document(
                upload.name, upload.getvalue(), on_conflict=policy
            )
        except (documents.DocumentError, OSError) as error:
            errors.append(f"**{upload.name}**: {error}")

    # Shown after the rerun below; a rerun would wipe anything drawn now.
    st.session_state.save_errors = errors

    # Nothing new reached the library, so there is nothing to index.
    if len(errors) < len(ready):
        st.session_state.index_report = run_indexing()

    # A new key gives an empty uploader on the next run.
    st.session_state.uploader_version += 1
    st.rerun()


def render_documents():
    backend = load_backend()

    st.title("📄 Documents")
    st.caption(
        "Upload files, then index them. Unchanged documents are detected "
        "with a content fingerprint and are never re-indexed."
    )

    for message in st.session_state.pop("save_errors", []):
        st.error(f"Not saved. {message}")

    if "index_report" in st.session_state:
        render_index_report(st.session_state.pop("index_report"))

    ready = render_upload_section()

    if st.button(
        "Save and index",
        type="primary",
        disabled=not ready,
        help="Saves the files above to the library and indexes them.",
    ):
        save_and_index(ready)

    st.subheader("Document library")

    library = backend.documents.list_documents()

    if not library:
        st.info("No documents yet. Upload some files above.")
    else:
        st.dataframe(
            [
                {
                    "Document": row["document_id"],
                    "Type": row["type"],
                    "Size": format_size(row["size_kb"]),
                    "Chunks": row["chunks"],
                    "Status": row["status"],
                }
                for row in library
            ],
            hide_index=True,
            width="stretch",
        )

        pending = [r for r in library if r["status"] == "Not indexed"]

        if pending:
            st.warning(
                f"{plural(len(pending), 'document')} not indexed yet. "
                "Use the button below."
            )

    if st.button(
        "Re-scan and index library",
        help="Indexes new or changed files in docs/ (for example files you "
        "copied there by hand), skips unchanged ones, and removes index "
        "entries for files that no longer exist.",
    ):
        st.session_state.index_report = run_indexing()
        st.rerun()


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

def render_settings():
    backend = load_backend()
    config = backend.config

    st.title("⚙️ Settings and system information")
    st.caption(
        "Read-only. To change these values edit backend/config.py and "
        "restart the app."
    )

    status = ollama_status()

    st.subheader("Ollama")

    if not status["reachable"]:
        st.error(
            f"Ollama is not reachable at {config.OLLAMA_HOST}. Start the "
            "Ollama app or run `ollama serve`."
        )
    else:
        processor = status["processor"] or (
            "not loaded yet (it loads into memory on the first question)"
        )

        key_value_table({
            "Host": config.OLLAMA_HOST,
            "Configured model": config.LLM_MODEL,
            "Model installed": "yes" if status["model_installed"] else "NO",
            "Running on": processor,
            "Installed models": ", ".join(status["installed_models"]) or "none",
        })

        if not status["model_installed"]:
            st.error(f"Run `ollama pull {config.LLM_MODEL}` in a terminal.")

    st.subheader("Models and retrieval")

    key_value_table({
        "Embedding model": config.EMBEDDING_MODEL,
        "Embedding dimension": config.EMBEDDING_DIMENSION,
        "Embedding device": backend.embeddings.get_embedding_device(),
        "Retrieval method": (
            f"Hybrid: ChromaDB semantic search + BM25, fused with "
            f"Reciprocal Rank Fusion (k={backend.hybrid.RRF_K})"
        ),
        "Top K": config.TOP_K,
        "Maximum cosine distance": config.MAX_DISTANCE,
        "Chunk size / overlap": f"{config.CHUNK_SIZE} / {config.CHUNK_OVERLAP} words",
    })

    st.subheader("Storage")

    key_value_table({
        "Vector database": str(config.DB_DIR),
        "Collection": f"{config.COLLECTION_NAME} ({config.DISTANCE_METRIC})",
        "Chunks stored": backend.store.get_collection_count(),
        "Document folder": str(config.DOCS_DIR),
        "Upload folder": str(config.UPLOAD_DIR),
        "Supported formats": ", ".join(sorted(config.SUPPORTED_EXTENSIONS)),
    })

    st.caption(
        "Chunk size, overlap and the embedding model are part of every "
        "document's fingerprint. Changing them re-indexes all documents on "
        "the next indexing run. A model with a different embedding "
        "dimension also needs a new collection name."
    )


# ---------------------------------------------------------------------------
# App shell
# ---------------------------------------------------------------------------

def render_startup_error(error: Exception):
    st.title("🔎 LocalRAG")
    st.error("LocalRAG could not start.")

    if isinstance(error, OSError):
        st.markdown(
            "The embedding model is not available in the local cache. "
            "LocalRAG never downloads models while running. Connect to the "
            "internet **once** and run this in PowerShell from the project "
            "folder:"
        )
        st.code(
            ".\\.venv\\Scripts\\python.exe -c \"from sentence_transformers "
            "import SentenceTransformer; "
            "SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')\"",
            language="powershell",
        )
    else:
        st.markdown("The backend failed to load.")

    st.caption(f"Details: {type(error).__name__}: {str(error)[:300]}")


def main():
    st.set_page_config(
        page_title="LocalRAG", page_icon="🔎", layout="wide"
    )

    # Organic theme: shapes, paper grain, controls and responsive rules.
    st.markdown(
        f"<style>{(Path(__file__).parent / 'theme.css').read_text(encoding='utf-8')}</style>",
        unsafe_allow_html=True,
    )

    try:
        load_backend()
    except Exception as error:
        logger.exception("Backend failed to load")
        render_startup_error(error)
        st.stop()

    page = st.navigation([
        st.Page(render_dashboard, title="Dashboard", icon="🏠", default=True),
        st.Page(render_chat, title="Chat", icon="💬", url_path="chat"),
        st.Page(render_documents, title="Documents", icon="📄", url_path="documents"),
        st.Page(render_settings, title="Settings", icon="⚙️", url_path="settings"),
    ])

    render_sidebar_status()

    try:
        page.run()
    except Exception as error:
        logger.exception("Unhandled error in page")
        st.error(
            "Something unexpected went wrong. Reload the page and try "
            f"again. ({type(error).__name__}: {str(error)[:200]})"
        )


if __name__ == "__main__":
    main()
