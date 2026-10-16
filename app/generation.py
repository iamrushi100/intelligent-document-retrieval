"""Provider-independent generation interface and retrieval-to-answer flow."""
from typing import Protocol, runtime_checkable

from app.rag import RetrievedChunk, build_context, format_context


class TextGenerationProviderError(RuntimeError):
    """Safe public error used when a configured provider request fails."""


@runtime_checkable
class TextGenerator(Protocol):
    """Interface implemented by development and future provider generators."""

    def generate(self, question: str, context: str) -> str:
        """Return an answer using the question and formatted retrieved context."""
        ...


class MockTextGenerator:
    """Deterministic local generator; it never makes network requests."""

    def generate(self, question: str, context: str) -> str:
        question = question.strip()
        if not context.strip():
            return "No retrieved context was provided."
        return f"Mock answer for: {question}\n\nBased on the retrieved context:\n{context}"


class OpenAITextGenerator:
    """Generate grounded text with OpenAI's Responses API."""

    def __init__(self, api_key: str | None = None, model: str = "gpt-4o-mini", client=None):
        self.api_key = api_key
        self.model = model
        self._client = client

    def _get_client(self):
        if self._client is None:
            if not self.api_key:
                raise ValueError("OPENAI_API_KEY is required for OpenAI generation.")
            from openai import OpenAI

            self._client = OpenAI(api_key=self.api_key)
        return self._client

    def generate(self, question: str, context: str) -> str:
        try:
            response = self._get_client().responses.create(
                model=self.model,
                instructions=(
                    "Answer the question using only the supplied retrieved context. "
                    "If the context does not contain the answer, say that the available "
                    "documents do not provide enough information."
                ),
                input=f"Question:\n{question}\n\nRetrieved context:\n{context}",
            )
            return response.output_text
        except Exception:
            # Provider exceptions can contain request details. Keep them out of API output/logs.
            raise TextGenerationProviderError("Text generation provider request failed.") from None


class RagGenerationService:
    """Connect existing hybrid retrieval to a text generator while retaining sources."""

    def __init__(self, retriever, generator: TextGenerator):
        self.retriever = retriever
        self.generator = generator

    def answer(self, question: str, top_k: int = 5) -> dict:
        """Retrieve, format, and generate an answer with source metadata."""
        if not question.strip():
            raise ValueError("Question must not be empty.")

        results = self.retriever.search(question, top_k)
        chunks = build_context(results, top_k=top_k)
        context = format_context(chunks)
        sources = [_source_metadata(chunk) for chunk in chunks]
        if not chunks:
            return {
                "answer": "I could not find relevant documents to answer this question.",
                "sources": [],
            }
        return {"answer": self.generator.generate(question, context), "sources": sources}


def _source_metadata(chunk: RetrievedChunk) -> dict:
    return {
        "chunk_id": chunk.chunk_id,
        "document_id": chunk.document_id,
        "filename": chunk.filename,
        "retrieval_score": chunk.retrieval_score,
    }
