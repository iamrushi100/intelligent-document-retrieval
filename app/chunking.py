"""Split parsed text into overlapping chunks with source metadata."""
from app.models import DocumentChunk, ParsedPage

def _split_text(text: str, size: int, overlap: int) -> list[str]:
    """Prefer paragraph boundaries; split long paragraphs at character edges."""
    paragraphs = text.split("\n\n")
    pieces: list[str] = []
    current = ""
    for paragraph in paragraphs:
        while len(paragraph) > size:
            if current:
                pieces.append(current)
                current = ""
            pieces.append(paragraph[:size])
            paragraph = paragraph[size:]
        candidate = f"{current}\n\n{paragraph}" if current else paragraph
        if len(candidate) > size:
            pieces.append(current)
            current = paragraph
        else:
            current = candidate
    if current:
        pieces.append(current)
    if overlap == 0:
        return pieces
    return [piece if i == 0 else pieces[i - 1][-overlap:] + "\n" + piece
            for i, piece in enumerate(pieces)]

def create_chunks(pages: list[ParsedPage], document_id: str, document_name: str,
                  chunk_size: int, chunk_overlap: int) -> list[DocumentChunk]:
    """Create stable per-document chunk IDs and retain reliable page numbers."""
    if chunk_size < 1 or chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_size must be positive and overlap must be smaller than chunk_size")
    chunks = []
    for page in pages:
        for text in _split_text(page.text, chunk_size, chunk_overlap):
            chunks.append(DocumentChunk(f"{document_id}:{len(chunks)}", document_id,
                                        document_name, text, page.page_number))
    return chunks
