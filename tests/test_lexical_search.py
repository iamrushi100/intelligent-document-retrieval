from datetime import datetime, timezone

from fastapi.testclient import TestClient

from app.config import Settings
from app.lexical_search import LexicalSearchService
from app.main import create_app
from app.models import DocumentChunk
from app.storage import DocumentStore


def make_document(document_id, name, chunk, sha):
    document = {"id": document_id, "name": name, "file_type": "txt", "size_bytes": len(chunk.text),
                "sha256": sha, "status": "processed", "created_at": datetime.now(timezone.utc).isoformat()}
    return document, [chunk]


def test_exact_identifier_and_technical_term_are_retrievable(tmp_path):
    store = DocumentStore(tmp_path / "terms.db")
    store.initialize()
    chunks = [
        DocumentChunk("a:0", "a", "auth.txt", "Gateway AUTH-9321 requires the XGBoost reranker.", 2, "Gateway"),
        DocumentChunk("b:0", "b", "kubernetes.txt", "The Kubernetes deployment uses service discovery and health probes."),
        DocumentChunk("c:0", "c", "postgres.txt", "PostgreSQL vacuum tuning improves database maintenance."),
        DocumentChunk("d:0", "d", "tls.txt", "TLS certificates secure internal service communication."),
    ]
    for index, chunk in enumerate(chunks):
        document, one_chunk = make_document(chunk.document_id, chunk.document_name, chunk, f"term-hash-{index}")
        store.save_document(document, one_chunk)
    service = LexicalSearchService(store)
    identifier_result = service.search("AUTH-9321", top_k=3)
    technical_result = service.search("XGBoost", top_k=3)
    assert identifier_result[0]["chunk_id"] == "a:0"
    assert technical_result[0]["chunk_id"] == "a:0"


def test_relevant_passage_ranks_first(tmp_path):
    store = DocumentStore(tmp_path / "ranking.db")
    store.initialize()
    passages = [
        DocumentChunk("target:0", "target", "retry-guide.txt",
                      "The client retries failed network requests using exponential backoff, increasing delay after each failure."),
        DocumentChunk("near:0", "near", "network-guide.txt",
                      "The network client sends requests promptly; backoff can be configured separately."),
        DocumentChunk("other:0", "other", "cache-guide.txt",
                      "The cache keeps static assets until the configured TTL expires."),
        DocumentChunk("db:0", "db", "database-guide.txt",
                      "Database replication copies committed records to a secondary server."),
    ]
    for index, chunk in enumerate(passages):
        document, one_chunk = make_document(chunk.document_id, chunk.document_name, chunk, f"rank-hash-{index}")
        store.save_document(document, one_chunk)
    results = LexicalSearchService(store).search("exponential backoff retries network requests", 3)
    assert results[0]["chunk_id"] == "target:0"
    assert results[0]["bm25_score"] > results[1]["bm25_score"]


def test_legacy_chunks_are_indexed_and_metadata_is_preserved(tmp_path):
    store = DocumentStore(tmp_path / "legacy.db")
    store.initialize()
    chunk = DocumentChunk("policy:0", "policy", "policy.pdf",
                          "Policy AUTH-RET-90 requires token rotation every 90 days.", 6, "Credential Rotation")
    document, one_chunk = make_document("policy", "policy.pdf", chunk, "legacy-hash")
    store.save_document(document, one_chunk)
    service = LexicalSearchService(store)
    result = service.search("AUTH-RET-90", top_k=1)[0]
    assert result["chunk_id"] == "policy:0"
    assert result["document_name"] == "policy.pdf"
    assert result["page_number"] == 6
    assert result["section"] == "Credential Rotation"
    assert len(store.get_lexical_chunks("simple-v1")) == 1


def test_lexical_upload_is_searchable_and_api_validates_inputs(tmp_path):
    client = TestClient(create_app(Settings(database_path=tmp_path / "api.db")))
    with client:
        uploaded = client.post("/documents", files={
            "file": ("release.txt", b"Build artifact REL-2026-17 uses protobuf serialization.", "text/plain")
        })
        assert uploaded.status_code == 201
        found = client.get("/search/lexical", params={"q": "REL-2026-17", "top_k": 5})
        assert found.status_code == 200
        assert found.json()["results"][0]["document_name"] == "release.txt"
        assert client.get("/search/lexical", params={"q": "   "}).status_code == 422
        assert client.get("/search/lexical", params={"q": "REL-2026-17", "top_k": 0}).status_code == 422
        assert client.get("/search/lexical", params={"q": "REL-2026-17", "top_k": 101}).status_code == 422


def test_empty_lexical_index_returns_no_results(tmp_path):
    client = TestClient(create_app(Settings(database_path=tmp_path / "empty.db")))
    with client:
        response = client.get("/search/lexical", params={"q": "AUTH-9321"})
    assert response.status_code == 200
    assert response.json()["results"] == []
