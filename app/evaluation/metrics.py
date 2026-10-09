"""Ranking metrics used by the retrieval evaluation harness."""
from collections.abc import Collection, Sequence


def _validate_k(k: int) -> None:
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive integer")


def _unique_ids(ranked_chunk_ids: Sequence[str]) -> list[str]:
    unique = []
    seen = set()
    for chunk_id in ranked_chunk_ids:
        if not isinstance(chunk_id, str) or not chunk_id.strip():
            raise ValueError("ranked chunk IDs must be non-empty strings")
        if chunk_id not in seen:
            seen.add(chunk_id)
            unique.append(chunk_id)
    return unique


def _relevant_ids(relevant_chunk_ids: Collection[str]) -> set[str]:
    relevant = set()
    for chunk_id in relevant_chunk_ids:
        if not isinstance(chunk_id, str) or not chunk_id.strip():
            raise ValueError("relevant chunk IDs must be non-empty strings")
        relevant.add(chunk_id)
    return relevant


def recall_at_k(ranked_chunk_ids: Sequence[str], relevant_chunk_ids: Collection[str], k: int) -> float:
    """Return relevant unique chunks found in the first K divided by all relevant chunks.

    For a query with no relevant chunks (an explicitly unanswerable case), recall is 0.
    """
    _validate_k(k)
    ranked = _unique_ids(ranked_chunk_ids)[:k]
    relevant = _relevant_ids(relevant_chunk_ids)
    if not relevant:
        return 0.0
    return len(set(ranked) & relevant) / len(relevant)


def precision_at_k(ranked_chunk_ids: Sequence[str], relevant_chunk_ids: Collection[str], k: int) -> float:
    """Return relevant unique results in the first K divided by K result slots.

    Missing result slots and unanswerable cases count as non-relevant (zero precision).
    """
    _validate_k(k)
    ranked = _unique_ids(ranked_chunk_ids)[:k]
    relevant = _relevant_ids(relevant_chunk_ids)
    return len(set(ranked) & relevant) / k


def reciprocal_rank_at_k(ranked_chunk_ids: Sequence[str], relevant_chunk_ids: Collection[str], k: int) -> float:
    """Return reciprocal rank of the first relevant unique chunk within K, or zero."""
    _validate_k(k)
    ranked = _unique_ids(ranked_chunk_ids)[:k]
    relevant = _relevant_ids(relevant_chunk_ids)
    for rank, chunk_id in enumerate(ranked, start=1):
        if chunk_id in relevant:
            return 1.0 / rank
    return 0.0


def calculate_metrics(ranked_chunk_ids: Sequence[str], relevant_chunk_ids: Collection[str], k: int) -> dict[str, float]:
    """Calculate the same three @K metrics for one retrieval result."""
    return {
        "recall": recall_at_k(ranked_chunk_ids, relevant_chunk_ids, k),
        "precision": precision_at_k(ranked_chunk_ids, relevant_chunk_ids, k),
        "mrr": reciprocal_rank_at_k(ranked_chunk_ids, relevant_chunk_ids, k),
    }
