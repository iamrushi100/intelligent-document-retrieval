from fastapi.testclient import TestClient
import pytest

from app.config import Settings
from app.hybrid_search import HybridSearchService, fuse_rankings
from app.main import create_app


def item(chunk_id, name=None, **scores):
    return {"document_name": name or f"{chunk_id}.txt", "chunk_id": chunk_id,
            "text": f"Text for {chunk_id}", "page_number": 3, "section": "Notes", **scores}


class FixedRetriever:
    def __init__(self, results):
        self.results = results
        self.calls = []

    def search(self, query, top_k):
        self.calls.append((query, top_k))
        return self.results[:top_k]


def test_rrf_formula_uses_one_based_ranks():
    results = fuse_rankings(
        [item("doc", similarity=0.99), item("other")],
        [item("third"), item("fourth"), item("doc", bm25_score=500)],
        top_k=1,
        rrf_constant=60,
    )
    assert results[0]["chunk_id"] == "doc"
    assert results[0]["semantic_rank"] == 1
    assert results[0]["lexical_rank"] == 3
    assert results[0]["hybrid_score"] == pytest.approx(1 / 61 + 1 / 63)


def test_fuses_rankings_and_breaks_ties_deterministically():
    semantic = [item("a"), item("b")]
    lexical = [item("b", bm25_score=1000), item("c")]
    results = fuse_rankings(semantic, lexical, top_k=3)
    assert [entry["chunk_id"] for entry in results] == ["b", "a", "c"]
    assert results[0]["hybrid_score"] == pytest.approx(1 / 61 + 1 / 62)

    tied = fuse_rankings([item("z")], [item("a")], top_k=2)
    assert [entry["chunk_id"] for entry in tied] == ["a", "z"]


def test_deduplicates_chunks_within_and_across_retrievers():
    results = fuse_rankings([item("same"), item("same")], [item("same")], top_k=5)
    assert len(results) == 1
    assert results[0]["semantic_rank"] == 1
    assert results[0]["lexical_rank"] == 1
    assert results[0]["hybrid_score"] == pytest.approx(2 / 61)


def test_missing_retriever_results_and_metadata_are_preserved():
    result = fuse_rankings([], [item("only", name="policy.pdf")], top_k=5)[0]
    assert result["document_name"] == "policy.pdf"
    assert result["chunk_id"] == "only"
    assert result["text"] == "Text for only"
    assert result["page_number"] == 3
    assert result["section"] == "Notes"
    assert result["semantic_rank"] is None
    assert result["lexical_rank"] == 1
    assert result["hybrid_score"] == pytest.approx(1 / 61)


def test_service_calls_both_retrievers_with_requested_top_k():
    semantic = FixedRetriever([item("s")])
    lexical = FixedRetriever([item("l")])
    service = HybridSearchService(semantic, lexical)
    service.search("question", top_k=7)
    assert semantic.calls == [("question", 7)]
    assert lexical.calls == [("question", 7)]


def test_hybrid_endpoint_validates_inputs_and_handles_empty_indexes(tmp_path):
    client = TestClient(create_app(Settings(database_path=tmp_path / "empty.db")))
    with client:
        assert client.get("/search/hybrid", params={"q": "   "}).status_code == 422
        assert client.get("/search/hybrid", params={"q": "question", "top_k": 0}).status_code == 422
        assert client.get("/search/hybrid", params={"q": "question", "top_k": 101}).status_code == 422
        response = client.get("/search/hybrid", params={"q": "question"})
    assert response.status_code == 200
    assert response.json()["rrf_constant"] == 60
    assert response.json()["results"] == []


def test_hybrid_endpoint_reuses_real_search_services(tmp_path):
    client = TestClient(create_app(Settings(database_path=tmp_path / "integrated.db")))
    with client:
        upload = client.post("/documents", files={
            "file": ("policy.txt", b"Release AUTH-818 rotates credentials every 30 days.", "text/plain")
        })
        assert upload.status_code == 201
        response = client.get("/search/hybrid", params={"q": "AUTH-818 credential rotation", "top_k": 5})
    assert response.status_code == 200
    result = response.json()["results"][0]
    assert result["document_name"] == "policy.txt"
    assert result["chunk_id"].startswith(upload.json()["id"] + ":")
    assert result["semantic_rank"] == 1
    assert result["lexical_rank"] == 1
    assert result["page_number"] is None
    assert result["section"] is None
