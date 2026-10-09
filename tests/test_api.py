from io import BytesIO
from fastapi.testclient import TestClient
from app.config import Settings
from app.main import create_app

def client_for(tmp_path, **overrides):
    return TestClient(create_app(Settings(database_path=tmp_path / "test.db", **overrides)))

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
