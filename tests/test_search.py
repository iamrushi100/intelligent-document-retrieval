from datetime import datetime, timezone

from fastapi.testclient import TestClient

from app.config import Settings
from app.embeddings import MODEL_NAME
from app.main import create_app
from app.models import DocumentChunk
from app.semantic_search import SemanticSearchService
from app.storage import DocumentStore


def test_semantic_search_ranks_relevant_document_first(tmp_path):
    client = TestClient(create_app(Settings(database_path=tmp_path / "rank.db")))
    with client:
        unrelated = client.post("/documents", files={"file": ("space.txt", b"A satellite orbits a distant planet and measures starlight.", "text/plain")})
        relevant = client.post("/documents", files={"file": ("cats.txt", b"Domestic cats purr softly and meow to communicate with people.", "text/plain")})
        assert unrelated.status_code == relevant.status_code == 201
        response = client.get("/search", params={"q": "Which pet purrs and meows?", "top_k": 2})
    assert response.status_code == 200
    results = response.json()["results"]
    assert len(results) == 2
    assert results[0]["document_name"] == "cats.txt"
    assert results[0]["chunk_id"].startswith(relevant.json()["id"] + ":")
    assert isinstance(results[0]["similarity"], float)


def test_search_preserves_page_and_section_metadata(tmp_path):
    store = DocumentStore(tmp_path / "metadata.db")
    store.initialize()
    chunk = DocumentChunk("policy:0", "policy", "policy.pdf",
                          "Passwords must be rotated every 90 days.", 4, "Password Policy")
    document = {"id": "policy", "name": "policy.pdf", "file_type": "pdf",
                "size_bytes": 50, "sha256": "metadata-test-hash", "status": "processed",
                "created_at": datetime.now(timezone.utc).isoformat()}
    service = SemanticSearchService(store)
    store.save_document(document, [chunk])
    result = service.search("How often should passwords change?", top_k=1)[0]
    assert result["page_number"] == 4
    assert result["section"] == "Password Policy"
    assert result["chunk_id"] == "policy:0"


def test_search_validates_query_and_top_k_and_handles_empty_index(tmp_path):
    client = TestClient(create_app(Settings(database_path=tmp_path / "empty.db")))
    with client:
        assert client.get("/search", params={"q": "   "}).status_code == 422
        assert client.get("/search", params={"q": "question", "top_k": 0}).status_code == 422
        assert client.get("/search", params={"q": "question", "top_k": 101}).status_code == 422
        response = client.get("/search", params={"q": "question"})
    assert response.status_code == 200
    assert response.json()["results"] == []
