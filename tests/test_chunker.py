
from backend.chunker import split_into_chunks
from backend.config import CHUNK_SIZE, CHUNK_OVERLAP


def test_chunker_returns_chunks():
    text = " ".join(["word"] * 400)

    chunks = split_into_chunks(text)

    assert len(chunks) > 1


def test_chunk_size_is_respected():
    text = " ".join(["word"] * 400)

    chunks = split_into_chunks(text)

    for chunk in chunks:
        assert len(chunk.split()) <= CHUNK_SIZE


def test_chunk_overlap_is_correct():
    words = [f"word{i}" for i in range(400)]
    text = " ".join(words)

    chunks = split_into_chunks(text)

    first_chunk_words = chunks[0].split()
    second_chunk_words = chunks[1].split()

    assert first_chunk_words[-CHUNK_OVERLAP:] == (
        second_chunk_words[:CHUNK_OVERLAP]
    )


def test_empty_text_returns_no_chunks():
    chunks = split_into_chunks("")

    assert chunks == []


def test_whitespace_only_text_returns_no_chunks():
    chunks = split_into_chunks("   \n\t  ")

    assert chunks == []
