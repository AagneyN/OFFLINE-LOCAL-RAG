from sentence_transformers import SentenceTransformer

from backend.config import (
    EMBEDDING_MODEL,
    EMBEDDING_DIMENSION
)


# Load the embedding model once
_model = SentenceTransformer(
    EMBEDDING_MODEL,
    local_files_only=True
)


def generate_embeddings(texts: list[str]) -> list[list[float]]:
    """
    Convert a list of text passages into numerical vectors.
    """

    if not texts:
        return []

    embeddings = _model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    return embeddings.tolist()


def generate_embedding(text: str) -> list[float]:
    """
    Convert a single text into a numerical vector.
    """

    return generate_embeddings([text])[0]


def get_embedding_dimension() -> int:
    """Return the embedding vector dimension."""

    return EMBEDDING_DIMENSION


def get_embedding_device() -> str:
    """Return the device the embedding model actually runs on."""

    return str(_model.device)