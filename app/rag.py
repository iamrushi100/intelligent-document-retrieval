"""Small, framework-independent helpers for preparing retrieved context."""
from dataclasses import dataclass
from math import isfinite
from collections.abc import Sequence


@dataclass(frozen=True)
class RetrievedChunk:
    """The source fields needed to cite and pass one result to a generator."""

    chunk_id: str
    document_id: str | None
    filename: str
    text: str
    retrieval_score: float


_SCORE_FIELDS = ("hybrid_score", "similarity", "bm25_score")


def build_context(results: Sequence[dict], top_k: int = 5) -> list[RetrievedChunk]:
    """Normalize the first top-K retrieval results, preserving their ranking order."""
    if top_k < 1:
        raise ValueError("top_k must be positive.")

    selected = []
    for result in results[:top_k]:
        score_field = next((field for field in _SCORE_FIELDS if field in result), None)
        if score_field is None:
            raise ValueError("Retrieval result must include hybrid_score, similarity, or bm25_score.")
        score = float(result[score_field])
        if not isfinite(score):
            raise ValueError("Retrieval score must be finite.")
        selected.append(RetrievedChunk(
            chunk_id=str(result["chunk_id"]),
            document_id=result.get("document_id"),
            filename=str(result.get("document_name", result.get("filename", "unknown"))),
            text=str(result["text"]),
            retrieval_score=score,
        ))
    return selected


def format_context(chunks: Sequence[RetrievedChunk]) -> str:
    """Format normalized chunks as explicit, source-labeled LLM context."""
    sections = []
    for index, chunk in enumerate(chunks, start=1):
        sections.append(
            f"[Source {index}]\n"
            f"Filename: {chunk.filename}\n"
            f"Document ID: {chunk.document_id or 'unknown'}\n"
            f"Chunk ID: {chunk.chunk_id}\n"
            f"Retrieval score: {chunk.retrieval_score:.6f}\n"
            f"Text:\n{chunk.text}"
        )
    return "\n\n".join(sections)
