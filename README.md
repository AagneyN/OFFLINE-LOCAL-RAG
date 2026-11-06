# LocalRAG

A retrieval-augmented generation (RAG) application that runs **entirely on your own computer**.
Upload TXT and PDF documents, ask questions in a chat interface, and get answers that are
grounded in those documents — with the source passages shown next to every answer.

There are no API keys, no paid services and no cloud calls. Embeddings, vector search and the
language model all run locally.

- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` (runs on CPU)
- **Vector database:** ChromaDB (persistent, on disk)
- **Retrieval:** hybrid — semantic search + BM25 keyword search, combined with Reciprocal Rank Fusion
- **Language model:** `qwen2.5:3b-instruct` served by [Ollama](https://ollama.com)
- **Interface:** Streamlit

---

## Features

- **Document upload** for TXT and PDF files, with validation (unsupported types, empty files,
  corrupted or password-protected PDFs, scanned PDFs without text, non-UTF-8 text) and clear messages.
- **Safe re-indexing.** Every document gets a SHA-256 fingerprint. Unchanged documents are skipped.
  When a document changes, the new version is inserted *before* the old one is removed, so a failure
  half-way never leaves you with a missing document.
- **Duplicate and update handling.** Re-uploading an identical file does nothing. A different file
  with the same name asks whether to *keep both* (default) or *replace* the existing file.
  Uploads go into `docs/uploads/` and can never overwrite the bundled sample documents.
- **Hybrid retrieval** (semantic + BM25 + RRF) with per-source scores shown in the chat.
- **Answers that stay inside your documents.** Two safeguards work together. If no retrieved passage
  is close enough (cosine distance above `0.75`), LocalRAG says it cannot find sufficient
  information and the language model is not called at all. Otherwise the model must first *quote*
  the sentences that answer the question, and the code checks that those quotes really appear in
  the passages. An answer whose quotes are not found is replaced by "I don't have enough
  information in the documents to answer that." This is what stops a small model from filling
  gaps from its own memory (see the evaluation below).
- **Source attribution:** document, chunk number, cosine distance, RRF score, BM25 score and the
  chunk text for every answer. If the model declined, the passages it looked at are still shown,
  under the heading "Passages checked".
- **Dashboard and settings pages** showing models, configuration, document/chunk counts and whether
  Ollama and the configured model are available.
- **Readable errors** for the situations most likely to go wrong: Ollama not running, model not
  installed, embedding model not cached, unreadable documents, indexing failures.
- **Private by default:** the web interface only listens on `localhost`, and telemetry in both
  Streamlit and ChromaDB is switched off.

---

## How it works

```
                 INDEXING                                         QUESTION ANSWERING
                                                  
 TXT / PDF ──► load_document() ──► split_into_chunks()      question ──► embed query (MiniLM)
 (docs/)       PyMuPDF / UTF-8      180 words, 30 overlap                      │
                                          │                     ┌────────────┴───────────────┐
                                          ▼                     ▼                            ▼
                                  generate_embeddings()   semantic search             BM25 keyword search
                                  384-dim, normalised     (ChromaDB, cosine)          (rank-bm25)
                                          │                     └────────────┬───────────────┘
                                          ▼                                  ▼
                                   ChromaDB collection         Reciprocal Rank Fusion (k = 60) → top 5
                                   "localrag" (on disk)                      │
                                                                             ▼
                                                              keep chunks with cosine distance ≤ 0.75
                                                                             │
                                                        none left? ──► "I could not find sufficient information"
                                                                             │ (the LLM is not called)
                                                                             ▼
                                                    add the chunk that follows each of the best 2 matches
                                                                             │
                                                                             ▼
                                                     Ollama (qwen2.5:3b-instruct): quote the answering
                                                     sentences, then answer from those quotes only
                                                                             │
                                                                             ▼
                                                     quotes found in the passages? ──no──► "I don't have enough
                                                                             │ yes            information in the documents"
                                                                             ▼
                                                                  answer + source passages
```

1. **Load and chunk.** Text is extracted (PyMuPDF for PDFs) and split into overlapping chunks of
   180 words with a 30-word overlap.
2. **Embed and store.** Each chunk is embedded locally into a normalised 384-dimension vector and
   stored in ChromaDB together with its document id, chunk number and document fingerprint.
3. **Retrieve.** A question is embedded and matched two ways: by meaning (cosine similarity in
   ChromaDB) and by exact words (BM25). The two rankings are merged with Reciprocal Rank Fusion,
   which rewards chunks that rank well in either list.
4. **Filter.** Only chunks whose cosine distance is at most `MAX_DISTANCE = 0.75` are kept.
   If nothing passes, the question is rejected without calling the language model. Distance alone
   cannot separate "answerable" from "unanswerable": in the evaluation, questions about topics the
   documents only touch on (for example "How does Redis persistence work?") were closer than some
   genuinely answerable ones. So this filter only removes clearly unrelated questions.
5. **Add continuation.** Chunks are cut at a fixed word count, so an answer can run across a chunk
   boundary (for example a list whose last two items are in the next chunk). The chunk that follows
   each of the best `ADJACENT_CHUNKS = 2` matches is added to the context.
6. **Generate with verification.** The model is asked for JSON: first the sentences, copied word
   for word from the numbered passages, that answer the question, then an answer written from them.
   The code checks each quote against the passages (ignoring case and whitespace, tolerating a
   small copying slip). If no quote is found in the passages, the answer is discarded. Temperature
   is 0 with a fixed seed, so the same question gives the same answer.
7. **Show sources.** The retrieved chunks, with their scores, are shown under the answer, followed
   by any continuation passages that were also given to the model.

The web interface (`frontend/app.py`) contains no RAG logic. It calls the same backend functions
as the command-line tools: `index_documents()`, `ask_question()` and the modules behind them.

---

## Project structure

```
LocalRAG/
├── backend/
│   ├── config.py             all settings (models, chunking, thresholds, paths)
│   ├── document_loader.py    load_document(): TXT and PDF text extraction
│   ├── chunker.py            split_into_chunks()
│   ├── embeddings.py         local sentence-transformers embeddings
│   ├── vector_store.py       persistent ChromaDB collection
│   ├── indexer.py            index_documents(): fingerprints, safe versioned re-indexing
│   ├── document_manager.py   upload validation, safe file naming, duplicates, library listing
│   ├── retriever.py          semantic retrieval
│   ├── bm25_retriever.py     BM25 keyword retrieval
│   ├── hybrid_retriever.py   retrieve_hybrid(): Reciprocal Rank Fusion
│   ├── generator.py          generate_answer(): quote-and-verify generation via Ollama, get_ollama_status()
│   └── rag_pipeline.py       ask_question(): retrieve -> filter -> add continuation -> generate -> sources
├── frontend/
│   └── app.py                Streamlit interface (Dashboard, Chat, Documents, Settings)
├── .streamlit/config.toml    UI settings: localhost only, telemetry off, upload limit
├── docs/                     sample documents (docs/uploads/ holds files uploaded in the UI)
├── data/chroma_db/           persistent vector database (created on first indexing)
├── evaluation/               retrieval evaluation scripts and the end-to-end answer-quality evaluation
├── scripts/diagnostics/      small scripts for inspecting chunks and retrieval results
├── tests/                    pytest suite
├── index_documents.py        command line: index everything in docs/
├── query_documents.py        command line: interactive question answering
├── requirements.txt
└── pytest.ini
```

---

## Requirements

Developed and tested on:

- Windows 11, Intel Core i7 (13th gen), 16 GB RAM
- NVIDIA RTX 3050 Laptop GPU (6 GB VRAM)
- Python 3.12

A GPU is **not required**. The embedding model is small and runs on the CPU. Ollama uses the GPU
automatically when it can: on the machine above it ran `qwen2.5:3b-instruct` (about 2.2 GB)
entirely on the GPU (`ollama ps` reported `100% GPU`). Without a GPU, Ollama falls back to the CPU
and answers are slower. The Settings page shows where the model is actually running.

---

## Installation (Windows PowerShell)

Run these commands from the project folder.

### 1. Python environment

```powershell
python --version                       # should print Python 3.12.x
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell refuses to run `Activate.ps1`, allow scripts for this window only and retry:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

### 2. Ollama and the language model

Install Ollama for Windows from <https://ollama.com/download>, then open a **new** PowerShell
window and download the model (about 1.9 GB):

```powershell
ollama --version
ollama pull qwen2.5:3b-instruct
ollama list                            # qwen2.5:3b-instruct should be listed
```

The Ollama app runs in the background and listens on `http://localhost:11434`.

### 3. Embedding model (one-time download)

LocalRAG loads the embedding model from the local cache only and never downloads anything while
running. Fetch it once while you have internet access:

```powershell
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"
```

### 4. Build the index for the sample documents (optional)

You can also do this from the web interface.

```powershell
python index_documents.py
```

---

## Running the application

```powershell
.\.venv\Scripts\Activate.ps1
python -m streamlit run frontend/app.py
```

The interface opens at <http://localhost:8501>. Use `python -m streamlit` rather than the bare
`streamlit` command: it always uses the interpreter of the active environment, whereas the
`streamlit.exe` launcher inside a virtual environment that has been moved or renamed can silently
fail to start.

The first page load takes a few seconds while the embedding model is loaded. The first question
after starting Ollama is slower too, because the language model has to be loaded into memory.

### Using LocalRAG

**Documents → upload and index**

1. Open **Documents** and drop TXT or PDF files onto the uploader.
2. Each file is checked immediately. Problems (empty file, corrupted PDF, no extractable text, …)
   are shown next to the file, and that file is skipped.
3. If a file with the same name but different content already exists, choose **Keep both** (default)
   or **Replace the existing file**.
4. Press **Save and index**. A progress bar shows each document being processed, followed by a summary.
5. The **Document library** table lists every document, its chunk count and whether it is indexed.

Files you copy into `docs/` by hand are picked up with **Re-scan and index library**. That button
also removes index entries for files that no longer exist, so deleting a document from disk and
re-scanning is how you remove it from the knowledge base.

**Chat → ask questions**

Type a question. The answer appears with a **Sources used** section listing each passage that was
given to the model, with its document, chunk number and scores. If the documents do not contain
enough information the answer says so. **Clear conversation** empties the history.

**Settings** shows the active models, retrieval method, thresholds, storage paths and whether
Ollama and the configured model are available.

### Command-line tools

```powershell
python index_documents.py              # index everything in docs/
python query_documents.py              # ask questions in the terminal
```

---

## Offline operation

LocalRAG works without an internet connection once these have been downloaded:

| What | How it is obtained | When |
|---|---|---|
| Python packages | `pip install -r requirements.txt` | once |
| Ollama | installer from ollama.com | once |
| Language model | `ollama pull qwen2.5:3b-instruct` | once |
| Embedding model | the one-line command in step 3 above | once |

If the embedding model is not in the local cache the app shows a start-up message with the
command to run; it does not try to download it.

**What was checked.** The server was started with every HTTP(S) proxy variable pointing at a dead
proxy, so any outbound web request would have failed. In that session a question about the sample
documents was answered, a new document was uploaded, embedded and indexed, and a question about it
was answered. While this ran, the server process's TCP connections were polled every 0.15 seconds:
the only connections seen were on the loopback interface (the app talking to Ollama on
`127.0.0.1:11434` and to the browser). Streamlit and ChromaDB telemetry are disabled, and the web
server only listens on `localhost`.

That is good evidence that nothing needs the internet at run time, but it is not the same as a
test on a machine that is actually disconnected, and very short-lived connections could slip
between polls.

---

## Evaluation

### Method

`evaluation/evaluation_dataset.py` holds 26 hand-written questions about the five sample
documents: **23 answerable** and **3 unanswerable** (capital of France, inventor of the telephone,
a stock price). Each answerable question is labelled with the 1–2 chunks, identified as
`(file, chunk number)`, that contain the answer.

For each question the retrieval method returns the top *K* chunks, and chunks with a cosine distance
above `0.75` are discarded.

- **Precision@K** = labelled chunks retrieved ÷ chunks retrieved
- **Recall@K** = labelled chunks retrieved ÷ labelled chunks
- **Rejection accuracy** = share of all 26 questions where "at least one chunk survived the filter"
  matches whether the question is answerable

Precision and recall are averaged over the 23 answerable questions.

```powershell
python -m evaluation.compare_retrieval_methods     # semantic vs BM25 vs hybrid, K = 3 and 5
python -m evaluation.analyze_hybrid_errors         # per-question comparison of semantic and hybrid
python -m evaluation.compare_top_k                 # semantic only, K = 3 and 5
python -m evaluation.evaluate_rag                  # semantic only, per-question detail
python -m evaluation.answer_quality --set dev      # end to end, see "Answer-quality evaluation"
```

### Results

Output of `evaluation.compare_retrieval_methods` on the sample corpus (5 documents, 38 chunks),
re-run while preparing this README:

| Method | K | Precision | Recall | Rejection accuracy |
|---|---:|---:|---:|---:|
| Semantic | 3 | 54.35% | 91.30% | 100.00% |
| BM25 | 3 | 42.03% | 95.65% | 88.46% |
| Hybrid | 3 | 56.52% | 95.65% | 100.00% |
| Semantic | 5 | 46.38% | 97.83% | 100.00% |
| BM25 | 5 | 25.22% | 95.65% | 88.46% |
| **Hybrid (used by the app, `TOP_K = 5`)** | **5** | **49.06%** | **97.83%** | **100.00%** |

The app originally used K = 3 (hybrid: 56.52% / 95.65% / 100%). It now uses K = 5: a little lower
precision for higher recall, which matters more here because the model reads every passage it is
given and can only answer from what was retrieved.

### How to read these numbers

- **Precision is low partly by construction.** Most questions have one or two labelled chunks (15
  have one, 8 have two) but up to three chunks are returned, so a perfect system cannot reach 100%
  whenever three chunks pass the filter. Neighbouring chunks overlap by 30 words and often hold
  related text that was not labelled, so "unlabelled" does not always mean "irrelevant".
- **Recall is the more informative figure here:** on average, about 96% of a question's labelled
  chunks are among the top three (98% among the top five).
- **Rejection accuracy (100%) comes from only three unanswerable questions, all of them far from
  the documents' topics.** The distance filter alone does not catch questions that are *close* to
  the topics but not answered; see the answer-quality evaluation below, which tests that directly.
- **These metrics measure retrieval, not answers.** Answer quality is evaluated separately below.
- **The labels are tied to the current chunking** (180 words, 30 overlap) **and to the five sample
  documents.** Changing the chunk settings invalidates the chunk numbers, and adding your own
  documents adds more chunks to compete with the labelled ones.

These are project-specific results on a small hand-labelled set. They are not a benchmark and say
nothing about accuracy on other documents.

### Answer-quality evaluation (end to end)

The retrieval metrics above cannot tell whether the *final answer* is right, or whether the model
invents one when the documents have none. `evaluation/answer_quality.py` runs the real pipeline
(retrieve, filter, generate) and checks:

- **Context hit:** the sentences needed to answer were in the context given to the model.
- **Answer ok:** the generated answer matches regular-expression checks for the key facts.
- **Refused:** for questions the documents do not answer, the question was rejected or the model
  declined instead of answering.

The checks do not depend on chunk numbers, so they stay valid if the chunking changes. Three
question sets are used (`evaluation/answer_quality_dataset.py`):

| Set | Answerable | Unanswerable | Purpose |
|---|---:|---:|---|
| dev | 24 | 11 | the set every design decision was made on |
| held-out | 24 | 7 | written before tuning, to check that gains carry over |
| final | 12 | 8 | written before tuning and not examined until the end |

Many of the unanswerable questions are deliberately close to the documents' topics ("How do I
install PostgreSQL on Windows?", "Who founded Redis?", "What is the default learning rate of the
Adam optimizer?"), because far-off questions are easy.

```powershell
python -m evaluation.answer_quality --set dev --details     # or: heldout, final, all
```

**Results.** "Previous" is the earlier design (the original prompt, 3 chunks, no verification),
measured with the same questions; that configuration is no longer in the code. "Current" is the
design described above.

| Set | Previous: answer ok | Current: answer ok | Previous: refused | Current: refused |
|---|---:|---:|---:|---:|
| dev (24 / 11) | 95.83% | 95.83% | 72.73% | **100%** |
| held-out (24 / 7) | 100% | 91.67% | 71.43% | **100%** |
| final (12 / 8) | 100% | 91.67% | 75.00% | **100%** |
| **all** | **59 / 60** | **56 / 60** | **19 / 26** | **26 / 26** |

Context hit is 100% on all three sets for the current design (91.67% on dev for the previous one).
Of the 26 unanswerable questions, 13 are rejected by the distance filter before the model is
called and 13 are declined by the model after the quote check.

**What this says, and what it does not.**

- The previous design invented answers for 7 of 26 unanswerable questions, among them install
  steps for MySQL and PostgreSQL and a description of Redis persistence, taken from the model's own
  memory although the documents say nothing about them. The current design declined all 26.
- The price is **3 more answerable questions missed out of 60** (about 5 percentage points), and
  questions take about 2.3 s on average instead of about 1.5 s, because the model also writes the
  quotes (and, for questions that reach the model, about 2.6 s). Whether that trade is right depends
  on how bad a confident wrong answer is for your use; for a question-answering tool over
  documents, it usually is worse than "I don't know".
- The answerable questions missed by the current design: one **wrongly declined** (the model
  returned no usable quotes although the passage was in the context), one where a **fabricated
  quote** was correctly rejected (the model "quoted" a SQL statement that is not in the documents,
  so the question was declined), one **incomplete list** (only scikit-learn was quoted when asked
  for the Python libraries), and one **borderline** answer (PCA was described, but not called
  dimensionality reduction). The previous design missed one on the same sets.
- Individual results move slightly between runs of the same pipeline (GPU arithmetic and prompt
  caching mean temperature 0 is not perfectly repeatable), so differences of one question are noise.
- The checks are automatic and crude, and the sets are small (91 questions in total, all about
  five short documents). They compare versions of this pipeline. They are not a measure of accuracy
  on other documents, and not a substitute for reading answers.

**Alternatives that were tried on the dev set and not adopted:**

| Approach | Answerable ok | Refused | Why not |
|---|---:|---:|---|
| A stricter prompt only | 79% | 100% | the model became timid and declined answerable questions |
| The same plus a refusal example | 83% | 100% | same |
| A separate "can this be answered?" check, then the normal prompt | roughly 50–70% | 100% | far too many false refusals |
| Weight semantic search more, or cut passages far from the best match | recall fell | — | lost correct passages |
| Tighten `MAX_DISTANCE` | — | — | near-topic unanswerable questions were *closer* than some answerable ones |

---

## Testing

```powershell
python -m pytest
```

148 tests: the original 39 backend tests plus tests for the upload/validation logic, the indexer's
progress and error isolation, the Ollama status check, quote verification and the refusal path,
neighbour expansion, the answer-quality dataset and scorer, and the Streamlit interface (run
headlessly with Streamlit's `AppTest`, with all backend calls stubbed and upload folders
redirected to a temporary directory, so the tests never touch your database or documents). The
tests do not call Ollama. `tests/conftest.py` stops the pipeline tests from reading the real
vector database.

---

## Configuration

All settings live in `backend/config.py`:

| Setting | Default |
|---|---|
| `LLM_MODEL` | `qwen2.5:3b-instruct` |
| `OLLAMA_HOST` | `http://localhost:11434` |
| `EMBEDDING_MODEL` / `EMBEDDING_DIMENSION` | `sentence-transformers/all-MiniLM-L6-v2` / `384` |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `180` / `30` words |
| `TOP_K` | `5` (chunks retrieved per question) |
| `MAX_DISTANCE` | `0.75` (cosine distance) |
| `ADJACENT_CHUNKS` | `2` (the following chunk is added for this many of the best matches) |
| `COLLECTION_NAME` | `localrag` |

Chunk size, overlap and the embedding model are part of each document's fingerprint, so changing
them re-indexes every document the next time you index. A model with a different embedding
dimension also needs a new `COLLECTION_NAME`. The Streamlit server settings (port, upload limit,
telemetry) are in `.streamlit/config.toml`.

---

## Troubleshooting

| Message | What to do |
|---|---|
| "Ollama is not running" / "cannot reach Ollama" | Start the Ollama app, or run `ollama serve` in a terminal. |
| "The language model … is not installed" | `ollama pull qwen2.5:3b-instruct` |
| "LocalRAG could not start" (embedding model) | Run the one-time download from installation step 3 while online. |
| The page loads slowly while Ollama is stopped | Windows takes a few seconds to give up on a closed port; start Ollama. |
| `streamlit` prints nothing / does not start | Use `python -m streamlit run frontend/app.py`. |
| Warnings about `ScriptRunContext` and no browser window | `frontend/app.py` was run with plain `python` (for example the editor's Run button). Start it with `python -m streamlit run frontend/app.py`. |
| PowerShell blocks `Activate.ps1` | `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` |

---

## Limitations

- **Small-scale by design.** The BM25 index is rebuilt from the whole collection for every
  question. That is instant for a few hundred chunks but would not scale to very large collections.
- **No conversation memory.** Each question is answered on its own; follow-ups such as "and what
  about the second one?" do not work. Chat history lives in the browser session and is lost on reload.
- **A 3B model can be wrong.** Check important answers against the sources shown. The quote check
  guarantees that an answer is backed by a sentence that really is in the passages, not that the
  answer is a good reading of it: in testing, an experimental variant of this approach produced a
  fluent answer that attached the definition of LEFT JOIN to INNER JOIN. When the model declines, the passages it
  looked at are shown under "Passages checked".
- **It sometimes declines a question it could answer.** About 5 percentage points of the answerable
  evaluation questions were lost this way, or to answers that were incomplete (a list with only one
  of its items) or phrased differently from what was asked. This is the cost of not letting the
  model fall back on its own memory. Rephrasing the question often helps.
- **Short, extractive answers.** Because answers are built from quoted sentences they are accurate
  but plain, and a question that needs several sentences from different places may get only some.
- **A fixed distance threshold.** `0.75` was chosen against the small evaluation set above, not
  tuned per corpus, and it only removes clearly unrelated questions. Questions that are close to
  the documents' topics but not answered by them are handled by the quote check instead.
- **Slower than a plain prompt.** About 2.3 s per question on the test machine, against about
  1.5 s before the quote check was added.
- **Formats:** TXT (UTF-8) and PDF with a text layer only. There is no OCR, so scanned PDFs are
  rejected, and tables and images inside PDFs are not interpreted.
- **Single user, no authentication.** The interface listens on `localhost` only. Do not expose it
  to a network.
- **No delete button.** Remove a document by deleting its file from `docs/` and using
  **Re-scan and index library**.
- **Evaluation scope.** See "How to read these numbers" above.

## Future improvements

- Stream answers token by token instead of waiting for the full response.
- Condense follow-up questions using the conversation history before retrieval.
- A cross-encoder re-ranker for the top candidates (it would need one more model downloaded for
  offline use).
- Retry once, or ask for a longer list of quotes, when the model returns none for a question that
  retrieval clearly matched, to recover some of the wrongly declined questions.
- Persist the BM25 index instead of rebuilding it per query.
- Delete and re-index individual documents from the interface.
- OCR for scanned PDFs.
- A larger evaluation set drawn from more varied documents, and human review of the answers.
- Edit the retrieval settings (top K, threshold) from the Settings page.
