"""FastAPI endpoints for health checks, ingestion, and retrieval."""
from datetime import datetime, timezone
from hashlib import sha256
import re
import sqlite3
import uuid

from fastapi import Depends, FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import JSONResponse

from app.chunking import create_chunks
from app.config import Settings
from app.embeddings import MODEL_NAME
from app.hybrid_search import HybridSearchService
from app.lexical_search import LexicalSearchService, TOKENIZER_VERSION
from app.parsing import DocumentParseError, extract_text
from app.semantic_search import SemanticSearchService
from app.storage import DocumentStore

ALLOWED_EXTENSIONS = {"pdf", "docx", "txt"}

def safe_filename(filename: str | None) -> str:
    """Keep a plain basename and remove characters unsafe for display/storage."""
    raw = (filename or "").replace("\\", "/").split("/")[-1]
    cleaned = re.sub(r"[^A-Za-z0-9._ -]", "_", raw).strip(" .")
    return cleaned or "document"

def search_parameters(q: str = Query(min_length=1, max_length=2000),
                     top_k: int = Query(default=5, ge=1, le=100)) -> tuple[str, int]:
    """Share query validation between semantic, lexical, and hybrid search."""
    if not q.strip():
        raise HTTPException(status_code=422, detail="Query must contain non-whitespace text.")
    return q, top_k

def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_environment()
    if settings.chunk_size < 1 or not 0 <= settings.chunk_overlap < settings.chunk_size:
        raise ValueError("Invalid chunk size or overlap settings")
    store = DocumentStore(settings.database_path)
    store.initialize()
    semantic_search = SemanticSearchService(store)
    lexical_search = LexicalSearchService(store)
    hybrid_search = HybridSearchService(semantic_search, lexical_search)
    app = FastAPI(title="Intelligent Document Retrieval API", version="0.4.0")
    app.state.document_store = store
    app.state.semantic_search = semantic_search
    app.state.lexical_search = lexical_search
    app.state.hybrid_search = hybrid_search

    @app.exception_handler(sqlite3.IntegrityError)
    async def integrity_error_handler(request: Request, exc: sqlite3.IntegrityError):
        detail = "This document has already been uploaded." if "sha256" in str(exc) else "The document could not be stored."
        return JSONResponse(status_code=409, content={"detail": detail})

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/documents", status_code=201)
    async def upload_document(file: UploadFile = File(...)):
        name = safe_filename(file.filename)
        extension = name.rsplit(".", 1)[-1].lower() if "." in name else ""
        if extension not in ALLOWED_EXTENSIONS:
            raise HTTPException(status_code=415, detail="Supported file types are PDF, DOCX, and TXT.")
        content = await file.read(settings.max_upload_bytes + 1)
        if not content:
            raise HTTPException(status_code=422, detail="The uploaded file is empty.")
        if len(content) > settings.max_upload_bytes:
            raise HTTPException(status_code=413, detail="The uploaded file exceeds the size limit.")
        try:
            pages = extract_text(name, content)
        except DocumentParseError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        document_id = str(uuid.uuid4())
        chunks = create_chunks(pages, document_id, name, settings.chunk_size, settings.chunk_overlap)
        document = {"id": document_id, "name": name, "file_type": extension,
                    "size_bytes": len(content), "sha256": sha256(content).hexdigest(),
                    "status": "processed", "created_at": datetime.now(timezone.utc).isoformat()}
        embeddings = semantic_search.prepare_embeddings(chunks)
        lexical_terms = lexical_search.prepare_terms(chunks)
        store.save_document(document, chunks, embeddings, MODEL_NAME,
                            lexical_terms, TOKENIZER_VERSION)
        return {**{key: value for key, value in document.items() if key != "sha256"},
                "chunk_count": len(chunks)}

    @app.get("/documents")
    def list_documents():
        return {"documents": store.list_documents()}

    @app.get("/documents/{document_id}")
    def document_detail(document_id: str):
        document = store.get_document(document_id)
        if document is None:
            raise HTTPException(status_code=404, detail="Document not found.")
        document.pop("sha256", None)
        return document

    @app.get("/search")
    def search_documents(params: tuple[str, int] = Depends(search_parameters)):
        query, top_k = params
        return {"query": query, "top_k": top_k,
                "results": semantic_search.search(query, top_k)}

    @app.get("/search/lexical")
    def lexical_search_documents(params: tuple[str, int] = Depends(search_parameters)):
        query, top_k = params
        return {"query": query, "top_k": top_k,
                "results": lexical_search.search(query, top_k)}

    @app.get("/search/hybrid")
    def hybrid_search_documents(params: tuple[str, int] = Depends(search_parameters)):
        query, top_k = params
        return {"query": query, "top_k": top_k,
                "rrf_constant": hybrid_search.rrf_constant,
                "results": hybrid_search.search(query, top_k)}

    return app

app = create_app()
