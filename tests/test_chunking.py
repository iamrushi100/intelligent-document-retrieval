from app.chunking import create_chunks
from app.models import ParsedPage

def test_chunks_keep_page_and_document_metadata():
    chunks = create_chunks([ParsedPage("alpha beta gamma", 3)], "doc", "notes.txt", 10, 2)
    assert chunks[0].document_id == "doc"
    assert chunks[0].document_name == "notes.txt"
    assert chunks[0].page_number == 3
    assert chunks[0].chunk_id == "doc:0"

def test_long_paragraph_is_split_with_overlap():
    chunks = create_chunks([ParsedPage("abcdefghij")], "doc", "a.txt", 6, 2)
    assert [chunk.text for chunk in chunks] == ["abcdef", "ef\nghij"]
