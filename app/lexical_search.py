"""Score stored chunk tokens with Okapi BM25."""
import json
import re
from rank_bm25 import BM25Okapi
from app.models import DocumentChunk
from app.storage import DocumentStore

TOKENIZER_VERSION = "simple-v1"
_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+(?:[._/-][A-Za-z0-9]+)*")

def tokenize(text: str) -> list[str]:
    """Lowercase words while keeping common technical-identifier separators."""
    return [token.lower() for token in _TOKEN_PATTERN.findall(text)]

class LexicalSearchService:
    """Maintain token rows and build a small in-memory BM25 scorer per search."""
    def __init__(self, store: DocumentStore):
        self.store = store

    def prepare_terms(self, chunks: list[DocumentChunk]) -> dict[str, str]:
        """Tokenize chunks before storage so upload commits their lexical data too."""
        return {chunk.chunk_id: json.dumps(tokenize(chunk.text)) for chunk in chunks}

    def _index_missing_chunks(self) -> None:
        missing = self.store.list_unindexed_lexical_chunks(TOKENIZER_VERSION)
        if missing:
            self.store.save_lexical_terms(TOKENIZER_VERSION, self.prepare_terms(missing))

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Return matching chunks ordered by computed BM25 score."""
        query_tokens = tokenize(query)
        if top_k < 1 or top_k > 100:
            raise ValueError("top_k must be between 1 and 100.")
        if not query_tokens:
            return []
        self._index_missing_chunks()
        rows = self.store.get_lexical_chunks(TOKENIZER_VERSION)
        if not rows:
            return []
        corpus = [json.loads(row["terms"]) for row in rows]
        if not any(corpus):
            return []
        scorer = BM25Okapi(corpus)
        scores = scorer.get_scores(query_tokens)
        query_term_set = set(query_tokens)
        matching = [i for i, terms in enumerate(corpus) if query_term_set.intersection(terms)]
        ranked = sorted(matching, key=lambda i: float(scores[i]), reverse=True)[:top_k]
        return [{
            "document_name": rows[i]["document_name"],
            "chunk_id": rows[i]["chunk_id"],
            "text": rows[i]["text"],
            "page_number": rows[i]["page_number"],
            "section": rows[i]["section"],
            "bm25_score": float(scores[i]),
        } for i in ranked]
