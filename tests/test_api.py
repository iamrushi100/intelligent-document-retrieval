from io import BytesIO
from fastapi.testclient import TestClient
from app.config import Settings
from app.main import create_app

def client_for(tmp_path, **overrides):
    return TestClient(create_app(Settings(database_path=tmp_path / "test.db", **overrides)))


class RecordingGenerator:
    def __init__(self):
        self.calls = []

    def generate(self, question, context):
        self.calls.append((question, context))
        return "A generated answer."

def test_health_upload_list_and_detail(tmp_path):
    with client_for(tmp_path) as client:
        assert client.get("/health").json() == {"status": "ok"}
        response = client.post("/documents", files={"file": ("hello.txt", b"Hello document", "text/plain")})
        assert response.status_code == 201
        uploaded = response.json()
        assert uploaded["chunk_count"] == 1
        assert uploaded["status"] == "processed"
        assert client.get("/documents").json()["documents"][0]["id"] == uploaded["id"]
        detail = client.get(f"/documents/{uploaded['id']}").json()
        assert detail["chunks"][0]["text"] == "Hello document"
        assert detail["chunks"][0]["page_number"] is None

def test_upload_validation_and_duplicate(tmp_path):
    with client_for(tmp_path, max_upload_bytes=10) as client:
        assert client.post("/documents", files={"file": ("x.exe", b"a")}).status_code == 415
        assert client.post("/documents", files={"file": ("x.txt", b"")}).status_code == 422
        assert client.post("/documents", files={"file": ("x.txt", b"too many bytes")}).status_code == 413
    with client_for(tmp_path) as client:
        payload = ("same.txt", BytesIO(b"duplicate text"), "text/plain")
        assert client.post("/documents", files={"file": payload}).status_code == 201
        assert client.post("/documents", files={"file": payload}).status_code == 409

def test_detail_returns_404_for_unknown_document(tmp_path):
    with client_for(tmp_path) as client:
        assert client.get("/documents/not-a-real-id").status_code == 404

def test_frontend_and_saved_reports_are_served(tmp_path):
    with client_for(tmp_path) as client:
        page = client.get("/")
        assert page.status_code == 200
        assert "Search your library" in page.text
        assert client.get("/static/style.css").status_code == 200
        assert client.get("/static/app.js").status_code == 200
        report = client.get("/evaluation/results/evaluation.json")
        assert report.status_code == 200
        assert report.json()["per_question"]
        assert client.get("/evaluation/results/per_question.csv").status_code == 200


def test_ask_runs_hybrid_rag_flow_and_returns_sources(tmp_path):
    client = client_for(tmp_path)
    results = [{
        "chunk_id": "doc-1:0", "document_id": "doc-1", "document_name": "policy.txt",
        "text": "Rotate credentials every month.", "hybrid_score": 0.03,
    }]
    client.app.state.hybrid_search.search = lambda question, top_k: results[:top_k]
    generator = RecordingGenerator()
    client.app.state.rag_generation.generator = generator

    with client:
        response = client.post("/ask", json={
            "question": "How often should credentials rotate?", "top_k": 3,
        })
        # Existing retrieval endpoints remain available alongside /ask.
        hybrid_response = client.get("/search/hybrid", params={"q": "rotation"})

    assert response.status_code == 200
    assert response.json() == {
        "question": "How often should credentials rotate?",
        "answer": "A generated answer.",
        "sources": [{
            "chunk_id": "doc-1:0", "document_id": "doc-1",
            "filename": "policy.txt", "retrieval_score": 0.03,
        }],
    }
    assert generator.calls[0][0] == "How often should credentials rotate?"
    assert "Rotate credentials every month." in generator.calls[0][1]
    assert hybrid_response.status_code == 200
    assert hybrid_response.json()["results"] == results


def test_ask_validates_question_and_top_k(tmp_path):
    with client_for(tmp_path) as client:
        assert client.post("/ask", json={"question": "   "}).status_code == 422
        assert client.post("/ask", json={"question": ""}).status_code == 422
        assert client.post("/ask", json={"question": "q", "top_k": 0}).status_code == 422
        assert client.post("/ask", json={"question": "q", "top_k": 101}).status_code == 422
        assert client.post("/ask", json={}).status_code == 422


def test_ask_returns_deterministic_no_context_response(tmp_path):
    client = client_for(tmp_path)
    client.app.state.hybrid_search.search = lambda question, top_k: []

    with client:
        response = client.post("/ask", json={"question": "Question with no matches"})

    assert response.status_code == 200
    assert response.json() == {
        "question": "Question with no matches",
        "answer": "I could not find relevant documents to answer this question.",
        "sources": [],
    }


def test_upload_then_ask_uses_real_local_retrieval_pipeline(tmp_path):
    with client_for(tmp_path) as client:
        upload = client.post("/documents", files={
            "file": (
                "archive_policy.txt",
                b"PX-417 archive retention is exactly 43 days. The archive is purged after that period.",
                "text/plain",
            )
        })
        assert upload.status_code == 201
        document_id = upload.json()["id"]

        response = client.post("/ask", json={
            "question": "For PX-417, how many days is the archive retention period?"
        })

    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"]
    assert "43 days" in payload["answer"]
    assert payload["sources"]
    source = payload["sources"][0]
    assert source["document_id"] == document_id
    assert source["filename"] == "archive_policy.txt"
    assert source["chunk_id"].startswith(f"{document_id}:")
    assert source["retrieval_score"] > 0


def test_ask_hides_provider_exception_details(tmp_path, monkeypatch):
    api_key = "test-key-that-must-not-leak"
    monkeypatch.setenv("OPENAI_API_KEY", api_key)
    client = client_for(tmp_path, text_generator_provider="openai")
    client.app.state.hybrid_search.search = lambda question, top_k: [{
        "chunk_id": "doc:0", "document_id": "doc", "document_name": "policy.txt",
        "text": "Policy evidence.", "hybrid_score": 0.03,
    }]

    class FailingResponses:
        def create(self, **kwargs):
            raise RuntimeError(f"Authorization Bearer {api_key}; internal transport details")

    generator = client.app.state.rag_generation.generator
    generator._client = type("FakeClient", (), {"responses": FailingResponses()})()

    with client:
        response = client.post("/ask", json={"question": "What does the policy say?"})

    assert response.status_code == 502
    assert response.json() == {"detail": "Text generation provider request failed."}
    assert api_key not in response.text
    assert "internal transport details" not in response.text
