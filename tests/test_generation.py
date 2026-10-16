from pathlib import Path
from app.config import Settings
from app.generation import MockTextGenerator, OpenAITextGenerator, RagGenerationService, TextGenerator
from app.main import create_text_generator


class FixedRetriever:
    def __init__(self, results):
        self.results = results
        self.calls = []

    def search(self, question, top_k):
        self.calls.append((question, top_k))
        return self.results[:top_k]


class RecordingGenerator:
    def __init__(self):
        self.calls = []

    def generate(self, question, context):
        self.calls.append((question, context))
        return "Generated test answer"


def sample_result():
    return {
        "chunk_id": "doc-1:0", "document_id": "doc-1",
        "document_name": "policy.txt", "text": "Rotate credentials every month.",
        "hybrid_score": 0.03,
    }


def test_mock_generator_is_deterministic_and_uses_question_and_context():
    generator = MockTextGenerator()
    first = generator.generate("When rotate?", "Monthly rotation policy.")
    assert first == generator.generate("When rotate?", "Monthly rotation policy.")
    assert "When rotate?" in first
    assert "Monthly rotation policy." in first
    assert generator.generate("Question", "") == "No retrieved context was provided."


def test_generator_implements_provider_independent_interface():
    generator = MockTextGenerator()
    assert callable(generator.generate)
    assert isinstance(generator, TextGenerator)


def test_readme_describes_openai_as_opt_in_and_mock_as_default():
    readme = Path(__file__).resolve().parents[1] / "README.md"
    text = readme.read_text(encoding="utf-8")

    assert "An opt-in OpenAI adapter is implemented" in text
    assert "/ask` defaults to the deterministic mock" in text
    assert "Real LLM answer generation, reranking, and deletion endpoints remain future milestones" not in text


def test_mock_generator_remains_default():
    assert isinstance(create_text_generator(Settings()), MockTextGenerator)


def test_openai_key_alone_does_not_enable_provider(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.delenv("TEXT_GENERATOR_PROVIDER", raising=False)
    settings = Settings.from_environment()

    assert settings.text_generator_provider == "mock"
    assert isinstance(create_text_generator(settings), MockTextGenerator)


def test_openai_provider_loads_key_and_model_from_configuration(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    generator = create_text_generator(Settings(
        text_generator_provider="openai", openai_model="configured-model",
    ))

    assert isinstance(generator, OpenAITextGenerator)
    assert generator.api_key == "test-key"
    assert generator.model == "configured-model"


def test_openai_adapter_uses_fake_client_and_returns_text_without_network():
    class FakeResponses:
        def __init__(self):
            self.calls = []

        def create(self, **kwargs):
            self.calls.append(kwargs)
            return type("FakeResponse", (), {"output_text": "Answer from the supplied context."})()

    responses = FakeResponses()
    client = type("FakeClient", (), {"responses": responses})()
    generator = OpenAITextGenerator(model="test-model", client=client)

    answer = generator.generate("What is the policy?", "Policy says 43 days.")

    assert answer == "Answer from the supplied context."
    assert responses.calls == [{
        "model": "test-model",
        "instructions": (
            "Answer the question using only the supplied retrieved context. "
            "If the context does not contain the answer, say that the available "
            "documents do not provide enough information."
        ),
        "input": "Question:\nWhat is the policy?\n\nRetrieved context:\nPolicy says 43 days.",
    }]


def test_rag_flow_passes_question_and_formatted_context_and_preserves_sources():
    retriever = FixedRetriever([sample_result()])
    generator = RecordingGenerator()
    service = RagGenerationService(retriever, generator)

    result = service.answer("How often should credentials rotate?", top_k=3)

    assert retriever.calls == [("How often should credentials rotate?", 3)]
    assert len(generator.calls) == 1
    question, context = generator.calls[0]
    assert question == "How often should credentials rotate?"
    assert "Filename: policy.txt" in context
    assert "Chunk ID: doc-1:0" in context
    assert "Rotate credentials every month." in context
    assert result == {
        "answer": "Generated test answer",
        "sources": [{
            "chunk_id": "doc-1:0", "document_id": "doc-1",
            "filename": "policy.txt", "retrieval_score": 0.03,
        }],
    }


def test_empty_retrieval_returns_no_context_answer_without_calling_generator():
    generator = RecordingGenerator()
    result = RagGenerationService(FixedRetriever([]), generator).answer("Unindexed question")

    assert result == {
        "answer": "I could not find relevant documents to answer this question.",
        "sources": [],
    }
    assert generator.calls == []
