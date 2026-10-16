import pytest

from app.rag import RetrievedChunk, build_context, format_context


def test_build_context_normalizes_hybrid_results_and_limits_top_k():
    results = [
        {"chunk_id": "c1", "document_id": "d1", "document_name": "policy.pdf",
         "text": "First passage", "hybrid_score": 0.04, "semantic_rank": 1},
        {"chunk_id": "c2", "document_id": "d2", "document_name": "guide.txt",
         "text": "Second passage", "hybrid_score": 0.02},
    ]

    assert build_context(results, top_k=1) == [
        RetrievedChunk("c1", "d1", "policy.pdf", "First passage", 0.04)
    ]


@pytest.mark.parametrize(("score_field", "score"), [
    ("similarity", 0.8), ("bm25_score", 1.25), ("hybrid_score", 0.03),
])
def test_build_context_uses_actual_retrieval_score_fields(score_field, score):
    result = {"chunk_id": "c", "document_id": "d", "document_name": "file.txt",
              "text": "Passage", score_field: score}
    assert build_context([result])[0].retrieval_score == score


def test_format_context_labels_sources_and_preserves_metadata():
    chunk = RetrievedChunk("c1", "d1", "policy.pdf", "Rotate credentials monthly.", 0.04)
    assert format_context([chunk]) == (
        "[Source 1]\nFilename: policy.pdf\nDocument ID: d1\nChunk ID: c1\n"
        "Retrieval score: 0.040000\nText:\nRotate credentials monthly."
    )


def test_build_context_rejects_missing_or_nonfinite_scores_and_invalid_top_k():
    with pytest.raises(ValueError, match="top_k"):
        build_context([], top_k=0)
    with pytest.raises(ValueError, match="score"):
        build_context([{"chunk_id": "c", "text": "x"}])
    with pytest.raises(ValueError, match="finite"):
        build_context([{"chunk_id": "c", "text": "x", "hybrid_score": float("nan")}])
