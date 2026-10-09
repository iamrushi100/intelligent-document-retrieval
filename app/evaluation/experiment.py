"""Evaluate fixed question cases and write traceable retrieval results."""
import csv
import json
from pathlib import Path
from statistics import fmean
from typing import Any

from app.evaluation.metrics import calculate_metrics

RETRIEVAL_METHODS = ("semantic", "bm25", "hybrid")


def _mean_metrics(records: list[dict[str, float]]) -> dict[str, float]:
    if not records:
        return {"recall": 0.0, "precision": 0.0, "mrr": 0.0}
    return {metric: fmean(record[metric] for record in records)
            for metric in ("recall", "precision", "mrr")}


def evaluate_cases(cases: list[dict[str, Any]], source_chunks: dict[str, list[str]],
                   retrievers: dict[str, Any], evaluation_ks: list[int],
                   candidate_depth: int) -> dict[str, Any]:
    """Run every question through every method with identical depth and K cutoffs."""
    if set(retrievers) != set(RETRIEVAL_METHODS):
        raise ValueError(f"retrievers must be exactly {RETRIEVAL_METHODS}")
    if not evaluation_ks or any(isinstance(k, bool) or not isinstance(k, int) or k < 1 for k in evaluation_ks):
        raise ValueError("evaluation_ks must contain positive integers")
    if len(set(evaluation_ks)) != len(evaluation_ks):
        raise ValueError("evaluation_ks must not contain duplicates")
    if candidate_depth < max(evaluation_ks) or candidate_depth > 100:
        raise ValueError("candidate_depth must be at least max(evaluation_ks) and no more than 100")

    per_question = []
    values = {method: {k: [] for k in evaluation_ks} for method in RETRIEVAL_METHODS}
    answerable_values = {method: {k: [] for k in evaluation_ks} for method in RETRIEVAL_METHODS}
    unanswerable_count = sum(not case["relevant_sources"] for case in cases)

    for case in cases:
        relevant_ids = set()
        for source_id in case["relevant_sources"]:
            if source_id not in source_chunks or not source_chunks[source_id]:
                raise ValueError(f"No chunk IDs were produced for labeled source {source_id!r}.")
            relevant_ids.update(source_chunks[source_id])

        method_results = {}
        for method in RETRIEVAL_METHODS:
            retrieved = retrievers[method].search(case["question"], candidate_depth)
            ranked_ids = [result["chunk_id"] for result in retrieved]
            metrics_by_k = {}
            for k in evaluation_ks:
                metrics = calculate_metrics(ranked_ids, relevant_ids, k)
                metrics_by_k[str(k)] = metrics
                values[method][k].append(metrics)
                if relevant_ids:
                    answerable_values[method][k].append(metrics)
            method_results[method] = {
                "retrieved": [
                    {"chunk_id": result["chunk_id"],
                     "document_name": result.get("document_name"),
                     "score": result.get("similarity", result.get("bm25_score", result.get("hybrid_score"))),
                     "semantic_rank": result.get("semantic_rank"),
                     "lexical_rank": result.get("lexical_rank")}
                    for result in retrieved
                ],
                "retrieved_chunk_ids": ranked_ids,
                "metrics_at_k": metrics_by_k,
            }
        per_question.append({
            "case_id": case["case_id"],
            "question": case["question"],
            "category": case["category"],
            "relevant_sources": case["relevant_sources"],
            "relevant_chunk_ids": sorted(relevant_ids),
            "relevance_note": case["relevance_note"],
            "methods": method_results,
        })

    aggregate = {}
    for method in RETRIEVAL_METHODS:
        aggregate[method] = {}
        for k in evaluation_ks:
            aggregate[method][str(k)] = {
                "case_count": len(cases),
                "answerable_case_count": len(cases) - unanswerable_count,
                "unanswerable_case_count": unanswerable_count,
                "all_cases_macro": _mean_metrics(values[method][k]),
                "answerable_only_macro": _mean_metrics(answerable_values[method][k]),
            }
    return {"per_question": per_question, "aggregate": aggregate}


def write_results(report: dict[str, Any], output_dir: Path) -> tuple[Path, Path]:
    """Write deterministic, reviewable JSON and CSV result artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "evaluation.json"
    csv_path = output_dir / "per_question.csv"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    metric_ks = sorted({int(k) for data in report["per_question"]
                        for method in data["methods"].values() for k in method["metrics_at_k"]})
    metric_columns = [f"{name}@{k}" for k in metric_ks for name in ("recall", "precision", "mrr")]
    columns = ["case_id", "question", "category", "relevance_note", "relevant_sources",
               "relevant_chunk_ids", "method", "retrieved_chunk_ids", "retrieved_scores"] + metric_columns
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for case in report["per_question"]:
            for method_name, result in case["methods"].items():
                row = {key: case[key] for key in ("case_id", "question", "category", "relevance_note")}
                row["relevant_sources"] = json.dumps(case["relevant_sources"], ensure_ascii=False)
                row["relevant_chunk_ids"] = json.dumps(case["relevant_chunk_ids"], ensure_ascii=False)
                row["method"] = method_name
                row["retrieved_chunk_ids"] = json.dumps(result["retrieved_chunk_ids"], ensure_ascii=False)
                row["retrieved_scores"] = json.dumps(result["retrieved"], ensure_ascii=False)
                for k, metrics in result["metrics_at_k"].items():
                    for name, value in metrics.items():
                        row[f"{name}@{k}"] = value
                writer.writerow(row)
    return json_path, csv_path
