"""Create vector embeddings and rank chunks with exact cosine similarity."""
import numpy as np

from app.embeddings import MODEL_NAME, get_embedding_model
from app.models import DocumentChunk
from app.storage import DocumentStore

class SemanticSearchService:
    """Keep model loading, index catch-up, and top-K ranking out of API routes."""
    def __init__(self, store: DocumentStore, model_provider=get_embedding_model):
        self.store = store
        self.model_provider = model_provider
        self._model = None

    def _get_model(self):
        if self._model is None:
            self._model = self.model_provider()
        return self._model

    def _encode(self, texts: list[str]) -> np.ndarray:
        vectors = self._get_model().encode(
            texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
        )
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        if vectors.shape[0] != len(texts) or vectors.shape[1] == 0:
            raise ValueError("The embedding model returned an unexpected vector shape.")
        if not np.isfinite(vectors).all():
            raise ValueError("The embedding model returned non-finite values.")
        return vectors

    def prepare_embeddings(self, chunks: list[DocumentChunk]) -> dict[str, bytes]:
        """Encode new chunks and serialize float32 vectors for SQLite storage."""
        if not chunks:
            return {}
        vectors = self._encode([chunk.text for chunk in chunks])
        return {chunk.chunk_id: vector.tobytes() for chunk, vector in zip(chunks, vectors)}

    def _index_missing_chunks(self) -> None:
        missing = self.store.list_unindexed_chunks(MODEL_NAME)
        if missing:
            self.store.save_embeddings(MODEL_NAME, self.prepare_embeddings(missing))

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Return the top matching stored chunks using normalized-vector dot products."""
        query = query.strip()
        if not query:
            raise ValueError("Query must not be empty.")
        if top_k < 1 or top_k > 100:
            raise ValueError("top_k must be between 1 and 100.")
        self._index_missing_chunks()
        rows = self.store.get_indexed_chunks(MODEL_NAME)
        if not rows:
            return []
        query_vector = self._encode([query])[0]
        matrix = np.stack([np.frombuffer(row["vector"], dtype=np.float32) for row in rows])
        if matrix.shape[1] != query_vector.shape[0]:
            raise ValueError("Stored vectors do not match the active embedding model.")
        scores = matrix @ query_vector
        order = np.argsort(-scores, kind="stable")[:top_k]
        return [{
            "document_name": rows[i]["document_name"],
            "chunk_id": rows[i]["chunk_id"],
            "text": rows[i]["text"],
            "page_number": rows[i]["page_number"],
            "section": rows[i]["section"],
            "similarity": float(scores[i]),
        } for i in order]
