"""Run the checked-in corpus and question set through all three retrievers."""
import argparse

import hashlib
import importlib.metadata

from pathlib import Path
import platform
import tempfile
from typing import Any

from app.chunking import create_chunks
from app.config import Settings
from app.embeddings import MODEL_NAME
from app.evaluation.dataset import load_dataset
from app.evaluation.experiment import evaluate_cases, write_results
from app.hybrid_search import DEFAULT_RRF_CONSTANT, HybridSearchService
from app.lexical_search import LexicalSearchService, TOKENIZER_VERSION
from app.parsing import extract_text
from app.semantic_search import SemanticSearchService
from app.storage import DocumentStore

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CORPUS = ROOT / "evaluation" / "corpus"
DEFAULT_DATASET = ROOT / "evaluation" / "dataset.json"
DEFAULT_OUTPUT = ROOT / "evaluation" / "results"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def build_corpus(store: DocumentStore, corpus_dir: Path, semantic: SemanticSearchService,
                 lexical: LexicalSearchService) -> tuple[dict[str, list[str]], list[dict[str, Any]]]:
    """Parse and chunk fixture files through the same production ingestion functions."""
    source_chunks: dict[str, list[str]] = {}
    corpus_manifest = []
    files = sorted(corpus_dir.glob("*.txt"), key=lambda path: path.name.casefold())
    if not files:
        raise ValueError(f"No .txt corpus documents found in {corpus_dir}")
    for path in files:
        source_id = path.stem
        if source_id in source_chunks:
            raise ValueError(f"Duplicate corpus source ID: {source_id}")
        content = path.read_bytes()
        pages = extract_text(path.name, content)
        document_id = f"eval-{source_id}"
        chunks = create_chunks(pages, document_id, path.name, CHUNK_SIZE, CHUNK_OVERLAP)
        if not chunks:
            raise ValueError(f"Corpus document {path.name} produced no chunks")
        document = {"id": document_id, "name": path.name, "file_type": "txt",
                    "size_bytes": len(content), "sha256": _sha256(content),
                    "status": "processed", "created_at": "evaluation-fixture"}
        embeddings = semantic.prepare_embeddings(chunks)
        lexical_terms = lexical.prepare_terms(chunks)
        store.save_document(document, chunks, embeddings, MODEL_NAME,
                            lexical_terms, TOKENIZER_VERSION)
        ids = [chunk.chunk_id for chunk in chunks]
        source_chunks[source_id] = ids
        corpus_manifest.append({"source_id": source_id, "filename": path.name,
                                "sha256": _sha256(content), "chunk_ids": ids})
    return source_chunks, corpus_manifest


def run_experiment(corpus_dir: Path = DEFAULT_CORPUS, dataset_path: Path = DEFAULT_DATASET,
                   output_dir: Path = DEFAULT_OUTPUT, evaluation_ks: list[int] | None = None,
                   candidate_depth: int = 10) -> dict[str, Any]:
    """Build an isolated fixture index, evaluate, and write reproducibility artifacts."""
    evaluation_ks = evaluation_ks or [1, 3, 5]
    dataset = load_dataset(dataset_path, {path.stem for path in corpus_dir.glob("*.txt")})
    with tempfile.TemporaryDirectory(prefix="document-retrieval-eval-") as temp_dir:
        store = DocumentStore(Path(temp_dir) / "evaluation.sqlite3")
        store.initialize()
        semantic = SemanticSearchService(store)
        lexical = LexicalSearchService(store)
        source_chunks, corpus_manifest = build_corpus(store, corpus_dir, semantic, lexical)
        if candidate_depth < max(evaluation_ks):
            raise ValueError("candidate_depth must be at least the largest evaluation K")
        retrievers = {
            "semantic": semantic,
            "bm25": lexical,
            "hybrid": HybridSearchService(semantic, lexical, DEFAULT_RRF_CONSTANT),
        }
        evaluated = evaluate_cases(dataset["cases"], source_chunks, retrievers,
                                   evaluation_ks, candidate_depth)

    loaded_model = semantic._get_model()
    first_module = getattr(loaded_model, "_first_module", lambda: None)()
    transformer = getattr(first_module, "auto_model", None)
    model_revision = getattr(getattr(transformer, "config", None), "_commit_hash", None)
    report = {
        "format_version": 1,
        "configuration": {
            "python_version": platform.python_version(),
            "embedding_model": MODEL_NAME,
            "embedding_model_revision": model_revision,
            "embedding_device": "cpu",
            "semantic_library_version": importlib.metadata.version("sentence-transformers"),
            "bm25_library_version": importlib.metadata.version("rank-bm25"),
            "tokenizer_version": TOKENIZER_VERSION,
            "chunk_size_characters": CHUNK_SIZE,
            "chunk_overlap_characters": CHUNK_OVERLAP,
            "candidate_depth_per_retriever": candidate_depth,
            "evaluation_ks": evaluation_ks,
            "rrf_constant": DEFAULT_RRF_CONSTANT,
            "relevance_mapping": "Each labeled source resolves to every chunk created from that corpus source.",
            "unanswerable_metrics": "No relevant chunks means recall, precision, and MRR are all zero; these cases are also reported separately from answerable-only macro averages.",
        },
        "inputs": {
            "dataset_path": str(dataset_path.name),
            "dataset_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
            "corpus": corpus_manifest,
            "corpus_chunk_count": sum(len(ids) for ids in source_chunks.values()),
        },
        **evaluated,
    }
    write_results(report, output_dir)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-depth", type=int, default=10,
                        help="same candidate count requested from each retriever (default: 10)")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 3, 5])
    args = parser.parse_args()
    report = run_experiment(output_dir=args.output_dir,
                            evaluation_ks=args.ks,
                            candidate_depth=args.candidate_depth)
    json_path = args.output_dir / "evaluation.json"
    csv_path = args.output_dir / "per_question.csv"
    print(f"Cases: {len(report['per_question'])}")
    print(f"Corpus chunks: {report['inputs']['corpus_chunk_count']}")
    print(f"Candidate depth per retriever: {report['configuration']['candidate_depth_per_retriever']}")
    print("Answerable-only macro metrics:")
    for method, by_k in report["aggregate"].items():
        k = str(max(report["configuration"]["evaluation_ks"]))
        metrics = by_k[k]["answerable_only_macro"]
        print(f"  {method} @{k}: Recall={metrics['recall']:.4f}, "
              f"Precision={metrics['precision']:.4f}, MRR={metrics['mrr']:.4f}")
    print(f"JSON: {json_path}")
    print(f"CSV: {csv_path}")


if __name__ == "__main__":
    main()
