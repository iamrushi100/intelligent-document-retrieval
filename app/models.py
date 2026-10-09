"""Data structures shared by parsing, chunking, and storage."""
from dataclasses import dataclass

@dataclass(frozen=True)
class ParsedPage:
    text: str
    page_number: int | None = None

@dataclass(frozen=True)
class DocumentChunk:
    chunk_id: str
    document_id: str
    document_name: str
    text: str
    page_number: int | None = None
    section: str | None = None
