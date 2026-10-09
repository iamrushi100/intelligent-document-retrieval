"""Combine semantic and lexical candidate rankings using Reciprocal Rank Fusion."""
from collections.abc import Sequence

DEFAULT_RRF_CONSTANT = 60


def fuse_rankings(
    semantic_results: Sequence[dict],
    lexical_results: Sequence[dict],
    top_k: int = 5,
    rrf_constant: int = DEFAULT_RRF_CONSTANT,
) -> list[dict]:
    """Fuse one-based ranks, keeping one result per chunk and deterministic ties."""
    if top_k < 1:
        raise ValueError("top_k must be positive.")
    if rrf_constant < 1:
        raise ValueError("rrf_constant must be positive.")

    candidates: dict[str, dict] = {}
    ranks: dict[str, dict[str, int]] = {}
    scores: dict[str, float] = {}

    for rank_name, results in (("semantic_rank", semantic_results),
                               ("lexical_rank", lexical_results)):
        seen_in_retriever: set[str] = set()
        for rank, result in enumerate(results, start=1):
            chunk_id = result["chunk_id"]
            if chunk_id in seen_in_retriever:
                continue
            seen_in_retriever.add(chunk_id)
            if chunk_id not in candidates:
                candidates[chunk_id] = {
                    key: result.get(key) for key in
                    ("document_name", "chunk_id", "text", "page_number", "section")
                }
                ranks[chunk_id] = {"semantic_rank": None, "lexical_rank": None}
                scores[chunk_id] = 0.0
            ranks[chunk_id][rank_name] = rank
            scores[chunk_id] += 1.0 / (rrf_constant + rank)

    ordered_ids = sorted(
        candidates,
        key=lambda chunk_id: (
            -scores[chunk_id],
            min(rank for rank in ranks[chunk_id].values() if rank is not None),
            chunk_id,
        ),
    )
    return [
        {**candidates[chunk_id], **ranks[chunk_id], "hybrid_score": scores[chunk_id]}
        for chunk_id in ordered_ids[:top_k]
    ]


class HybridSearchService:
    """Ask existing retrievers for rankings and merge them without mixing raw scores."""
    def __init__(self, semantic_search, lexical_search,
                 rrf_constant: int = DEFAULT_RRF_CONSTANT):
        if rrf_constant < 1:
            raise ValueError("rrf_constant must be positive.")
        self.semantic_search = semantic_search
        self.lexical_search = lexical_search
        self.rrf_constant = rrf_constant

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Retrieve top-K candidates from both baselines, then fuse their ranks."""
        if top_k < 1 or top_k > 100:
            raise ValueError("top_k must be between 1 and 100.")
        semantic_results = self.semantic_search.search(query, top_k)
        lexical_results = self.lexical_search.search(query, top_k)
        return fuse_rankings(semantic_results, lexical_results, top_k, self.rrf_constant)
