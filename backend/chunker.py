from backend.config import CHUNK_SIZE, CHUNK_OVERLAP


def split_into_chunks(text: str) -> list[str]:
    """
    Split text into overlapping word-based chunks.
    """

    words = text.split()

    chunks = []
    start = 0

    while start < len(words):

        end = min(start + CHUNK_SIZE, len(words))

        chunk = " ".join(words[start:end])

        if chunk:
            chunks.append(chunk)

        if end == len(words):
            break

        start = end - CHUNK_OVERLAP

    return chunks