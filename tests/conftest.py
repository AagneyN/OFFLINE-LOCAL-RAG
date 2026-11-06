import pytest

from backend import rag_pipeline


@pytest.fixture(autouse=True)
def no_neighbour_lookups(monkeypatch):
    """Keep the pipeline tests independent of the real vector database.

    ask_question() looks up the chunk that follows each top result. Without
    this, a test that returns a made-up result such as ("databases_sql.txt",
    chunk 0) would pick up the real chunk 1 from the user's database. Tests of
    that lookup replace this stub themselves.
    """

    monkeypatch.setattr(rag_pipeline, "get_chunk", lambda document_id, chunk_index: None)
