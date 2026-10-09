"""Validate and extract readable text from the supported file formats."""
from io import BytesIO
import re
import unicodedata
import zipfile
from docx import Document as DocxDocument
from pypdf import PdfReader
from app.models import ParsedPage

class DocumentParseError(ValueError):
    """Raised when a supported file is malformed or contains no usable text."""

def normalize_text(text: str) -> str:
    """Normalize line endings and whitespace while retaining paragraph breaks."""
    text = unicodedata.normalize("NFKC", text).replace("\x00", "")
    lines = [re.sub(r"[\t \f\v]+", " ", line).strip() for line in text.splitlines()]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()

def extract_text(filename: str, content: bytes) -> list[ParsedPage]:
    """Extract pages for PDFs and one unpaginated text unit for DOCX/TXT."""
    extension = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    try:
        if extension == "pdf":
            if not content.startswith(b"%PDF-"):
                raise DocumentParseError("The uploaded PDF is invalid.")
            reader = PdfReader(BytesIO(content), strict=True)
            pages = [ParsedPage(normalize_text(page.extract_text() or ""), i)
                     for i, page in enumerate(reader.pages, start=1)]
        elif extension == "docx":
            with zipfile.ZipFile(BytesIO(content)) as archive:
                if "word/document.xml" not in archive.namelist():
                    raise DocumentParseError("The uploaded DOCX is invalid.")
            document = DocxDocument(BytesIO(content))
            pages = [ParsedPage(normalize_text("\n\n".join(p.text for p in document.paragraphs)))]
        elif extension == "txt":
            pages = [ParsedPage(normalize_text(content.decode("utf-8-sig")))]
        else:
            raise DocumentParseError("Supported file types are PDF, DOCX, and TXT.")
    except DocumentParseError:
        raise
    except Exception as exc:
        raise DocumentParseError("The uploaded file could not be parsed.") from exc
    pages = [page for page in pages if page.text]
    if not pages:
        raise DocumentParseError("The document contains no extractable text.")
    return pages
