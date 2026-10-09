"""SQLite persistence for documents, chunks, semantic vectors, and lexical tokens."""
from contextlib import contextmanager
import sqlite3
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from app.models import DocumentChunk

class DocumentStore:
    """Own the SQLite schema and transaction boundaries for documents and indexes."""
    def __init__(self, database_path: Path):
        self.database_path = database_path

    def _connect(self) -> sqlite3.Connection:
        if str(self.database_path) != ":memory:":
            self.database_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(str(self.database_path))
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection


    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        """Commit successful operations and always release the SQLite file handle."""
        connection = self._connect()
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()
    def initialize(self) -> None:
        with self._connection() as connection:
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY, name TEXT NOT NULL, file_type TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL, sha256 TEXT NOT NULL UNIQUE,
                    status TEXT NOT NULL, created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY, document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                    document_name TEXT NOT NULL, text TEXT NOT NULL, page_number INTEGER, section TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id);
                CREATE TABLE IF NOT EXISTS chunk_embeddings (
                    chunk_id TEXT PRIMARY KEY REFERENCES chunks(id) ON DELETE CASCADE,
                    model_name TEXT NOT NULL, vector BLOB NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_embeddings_model ON chunk_embeddings(model_name);
                CREATE TABLE IF NOT EXISTS lexical_chunks (
                    chunk_id TEXT PRIMARY KEY REFERENCES chunks(id) ON DELETE CASCADE,
                    tokenizer_version TEXT NOT NULL, terms TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_lexical_tokenizer ON lexical_chunks(tokenizer_version);
            """)

    def save_document(self, document: dict[str, Any], chunks: list[DocumentChunk],
                      embeddings: dict[str, bytes] | None = None, model_name: str | None = None,
                      lexical_terms: dict[str, str] | None = None,
                      tokenizer_version: str | None = None) -> None:
        """Insert document data and any supplied semantic and lexical indexes atomically."""
        with self._connection() as connection:
            connection.execute("INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?, ?)",
                (document["id"], document["name"], document["file_type"], document["size_bytes"],
                 document["sha256"], document["status"], document["created_at"]))
            connection.executemany("INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?)",
                [(c.chunk_id, c.document_id, c.document_name, c.text, c.page_number, c.section)
                 for c in chunks])
            if embeddings:
                if model_name is None:
                    raise ValueError("model_name is required when saving embeddings")
                connection.executemany(
                    "INSERT INTO chunk_embeddings (chunk_id, model_name, vector) VALUES (?, ?, ?)",
                    [(chunk_id, model_name, vector) for chunk_id, vector in embeddings.items()])
            if lexical_terms:
                if tokenizer_version is None:
                    raise ValueError("tokenizer_version is required when saving lexical terms")
                connection.executemany(
                    "INSERT INTO lexical_chunks (chunk_id, tokenizer_version, terms) VALUES (?, ?, ?)",
                    [(chunk_id, tokenizer_version, terms) for chunk_id, terms in lexical_terms.items()])

    def save_embeddings(self, model_name: str, embeddings: dict[str, bytes]) -> None:
        """Insert or refresh vectors for chunks needing semantic indexing."""
        with self._connection() as connection:
            connection.executemany("""
                INSERT INTO chunk_embeddings (chunk_id, model_name, vector) VALUES (?, ?, ?)
                ON CONFLICT(chunk_id) DO UPDATE SET model_name = excluded.model_name, vector = excluded.vector
            """, [(chunk_id, model_name, vector) for chunk_id, vector in embeddings.items()])

    def list_unindexed_chunks(self, model_name: str) -> list[DocumentChunk]:
        """Return chunks without a vector for the requested model version."""
        with self._connection() as connection:
            rows = connection.execute("""
                SELECT c.id, c.document_id, c.document_name, c.text, c.page_number, c.section
                FROM chunks c LEFT JOIN chunk_embeddings e ON e.chunk_id = c.id AND e.model_name = ?
                WHERE e.chunk_id IS NULL ORDER BY c.rowid
            """, (model_name,)).fetchall()
        return [DocumentChunk(row["id"], row["document_id"], row["document_name"],
                              row["text"], row["page_number"], row["section"]) for row in rows]

    def get_indexed_chunks(self, model_name: str) -> list[dict[str, Any]]:
        """Load aligned source metadata and vectors for exact semantic scoring."""
        with self._connection() as connection:
            rows = connection.execute("""
                SELECT c.id AS chunk_id, c.document_name, c.text, c.page_number, c.section, e.vector
                FROM chunks c JOIN chunk_embeddings e ON e.chunk_id = c.id
                WHERE e.model_name = ? ORDER BY c.rowid
            """, (model_name,)).fetchall()
        return [dict(row) for row in rows]

    def save_lexical_terms(self, tokenizer_version: str, terms_by_chunk: dict[str, str]) -> None:
        """Insert or refresh tokenized text for chunks needing lexical indexing."""
        with self._connection() as connection:
            connection.executemany("""
                INSERT INTO lexical_chunks (chunk_id, tokenizer_version, terms) VALUES (?, ?, ?)
                ON CONFLICT(chunk_id) DO UPDATE SET tokenizer_version = excluded.tokenizer_version,
                    terms = excluded.terms
            """, [(chunk_id, tokenizer_version, terms)
                  for chunk_id, terms in terms_by_chunk.items()])

    def list_unindexed_lexical_chunks(self, tokenizer_version: str) -> list[DocumentChunk]:
        """Return chunks that lack token data for the active tokenizer version."""
        with self._connection() as connection:
            rows = connection.execute("""
                SELECT c.id, c.document_id, c.document_name, c.text, c.page_number, c.section
                FROM chunks c LEFT JOIN lexical_chunks l
                    ON l.chunk_id = c.id AND l.tokenizer_version = ?
                WHERE l.chunk_id IS NULL ORDER BY c.rowid
            """, (tokenizer_version,)).fetchall()
        return [DocumentChunk(row["id"], row["document_id"], row["document_name"],
                              row["text"], row["page_number"], row["section"]) for row in rows]

    def get_lexical_chunks(self, tokenizer_version: str) -> list[dict[str, Any]]:
        """Load token lists and matching source metadata for the BM25 corpus."""
        with self._connection() as connection:
            rows = connection.execute("""
                SELECT c.id AS chunk_id, c.document_name, c.text, c.page_number, c.section, l.terms
                FROM chunks c JOIN lexical_chunks l ON l.chunk_id = c.id
                WHERE l.tokenizer_version = ? ORDER BY c.rowid
            """, (tokenizer_version,)).fetchall()
        return [dict(row) for row in rows]

    def list_documents(self) -> list[dict[str, Any]]:
        with self._connection() as connection:
            rows = connection.execute("""
                SELECT d.*, COUNT(c.id) AS chunk_count FROM documents d
                LEFT JOIN chunks c ON c.document_id = d.id
                GROUP BY d.id ORDER BY d.created_at DESC
            """).fetchall()
        return [dict(row) for row in rows]

    def get_document(self, document_id: str) -> dict[str, Any] | None:
        with self._connection() as connection:
            row = connection.execute("SELECT * FROM documents WHERE id = ?", (document_id,)).fetchone()
            if row is None:
                return None
            result = dict(row)
            result["chunks"] = [dict(chunk) for chunk in connection.execute(
                "SELECT id AS chunk_id, document_name, text, page_number, section "
                "FROM chunks WHERE document_id = ? ORDER BY id", (document_id,)).fetchall()]
        return result
