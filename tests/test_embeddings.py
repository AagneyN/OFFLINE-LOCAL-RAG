
from backend.embeddings import (
    generate_embedding,
    generate_embeddings,
    get_embedding_dimension
)

from backend.config import EMBEDDING_DIMENSION


def test_single_embedding_dimension():
    embedding = generate_embedding("What is machine learning?")

    assert len(embedding) == EMBEDDING_DIMENSION


def test_multiple_embeddings():
    texts = [
        "What is artificial intelligence?",
        "What is machine learning?",
        "What is deep learning?"
    ]

    embeddings = generate_embeddings(texts)

    assert len(embeddings) == 3

    for embedding in embeddings:
        assert len(embedding) == EMBEDDING_DIMENSION


def test_empty_list_returns_empty_list():
    embeddings = generate_embeddings([])

    assert embeddings == []


def test_embedding_dimension_configuration():
    assert get_embedding_dimension() == EMBEDDING_DIMENSION


def test_embeddings_are_normalized():
    embedding = generate_embedding("What is a database?")

    magnitude = sum(value ** 2 for value in embedding) ** 0.5

    assert abs(magnitude - 1.0) < 0.001
