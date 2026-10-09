"""Load the local sentence-embedding model once per process."""
from functools import lru_cache

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

@lru_cache(maxsize=1)
def get_embedding_model():
    """Load a CPU model lazily; SentenceTransformer caches downloaded weights locally."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(MODEL_NAME, device="cpu")
