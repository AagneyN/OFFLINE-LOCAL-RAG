from pathlib import Path

#Project root directory

BASE_DIR = Path(__file__).resolve().parent.parent

# Directory Configurations

DOCS_DIR = BASE_DIR / "docs"
# Files uploaded through the web UI live in their own sub-folder so they can
# never overwrite the documents that ship with the project.
UPLOAD_DIR = DOCS_DIR / "uploads"
DATA_DIR = BASE_DIR / "data"
DB_DIR = DATA_DIR / "chroma_db"


#ChromaDB Configurations
COLLECTION_NAME = "localrag"
DISTANCE_METRIC = "cosine"

#Text Chunking Configurations
CHUNK_SIZE = 180
CHUNK_OVERLAP = 30

#Embedding Model Configurations
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384

#Text Chunking Configurations

LLM_MODEL = "qwen2.5:3b-instruct"
OLLAMA_HOST = "http://localhost:11434"

# Chunks retrieved per question. 5 rather than 3: on the answer-quality
# evaluation it finds the needed passage more often (see README).
TOP_K = 5
MAX_DISTANCE = 0.75

# For the best-ranked chunks the chunk that follows them in the same document
# is also given to the model, so an answer that runs across a chunk boundary
# (for example a list that is split in two) is not cut off.
ADJACENT_CHUNKS = 2

# Indexing configuration

SUPPORTED_EXTENSIONS = {".txt", ".pdf"} 