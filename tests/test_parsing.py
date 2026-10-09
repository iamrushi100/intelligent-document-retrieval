import pytest
from app.parsing import DocumentParseError, extract_text, normalize_text

def test_txt_normalization_keeps_paragraphs():
    assert normalize_text(" First   line\r\n\r\n Second\tline ") == "First line\n\nSecond line"
    assert extract_text("note.txt", b"Hello\nworld")[0].text == "Hello\nworld"

def test_invalid_pdf_and_empty_text_are_rejected():
    with pytest.raises(DocumentParseError):
        extract_text("bad.pdf", b"not a pdf")
    with pytest.raises(DocumentParseError):
        extract_text("empty.txt", b"  \n")
