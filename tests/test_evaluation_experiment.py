import csv
import json

import pytest

from app.evaluation.experiment import evaluate_cases, write_results


class FixedRetriever:
    def __init__(self, results):
        self.results = results
        self.calls = []

    def search(self, question, depth):
        self.calls.append((question, depth))
        return self.results


def sample_case():
    return {"case_id": "case-1", "question": "Where is policy?", "category": "ordinary_question",
            "relevant_sources": ["policy"], "relevance_note": "Policy chunk contains the answer."}


def make_retrievers():
    return {
        "semantic": FixedRetriever([{"chunk_id": "policy:0", "document_name": "policy.txt", "similarity": 0.8}]),
        "bm25": FixedRetriever([{"chunk_id": "other:0", "document_name": "other.txt", "bm25_score": 1.2}]),
        "hybrid": FixedRetriever([{"chunk_id": "policy:0", "document_name": "policy.txt", "hybrid_score": 0.03}]),
    }


def test_experiment_uses_same_candidate_depth_for_all_methods_and_records_results():
    retrievers = make_retrievers()
    result = evaluate_cases([sample_case()], {"policy": ["policy:0"]}, retrievers, [1, 3], 5)
    for retriever in retrievers.values():
        assert retriever.calls == [("Where is policy?", 5)]
    assert result["per_question"][0]["relevant_chunk_ids"] == ["policy:0"]
    assert set(result["aggregate"]) == {"semantic", "bm25", "hybrid"}


def test_experiment_rejects_depth_smaller_than_evaluation_k():
    with pytest.raises(ValueError):
        evaluate_cases([sample_case()], {"policy": ["policy:0"]}, make_retrievers(), [1, 5], 3)


def test_json_csv_outputs_are_reproducible_and_traceable(tmp_path):
    report = evaluate_cases([sample_case()], {"policy": ["policy:0"]}, make_retrievers(), [1, 3], 5)
    first = tmp_path / "first"
    second = tmp_path / "second"
    write_results(report, first)
    write_results(report, second)
    assert (first / "evaluation.json").read_bytes() == (second / "evaluation.json").read_bytes()
    assert (first / "per_question.csv").read_bytes() == (second / "per_question.csv").read_bytes()
    output = json.loads((first / "evaluation.json").read_text(encoding="utf-8"))
    assert output["per_question"][0]["methods"]["bm25"]["retrieved_chunk_ids"] == ["other:0"]
    with (first / "per_question.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert {row["method"] for row in rows} == {"semantic", "bm25", "hybrid"}

def test_unanswerable_cases_are_explicitly_zero_scored_and_counted():
    case = sample_case()
    case["category"] = "unanswerable"
    case["relevant_sources"] = []
    evaluated = evaluate_cases([case], {}, make_retrievers(), [1], 1)
    metrics = evaluated["per_question"][0]["methods"]["semantic"]["metrics_at_k"]["1"]
    aggregate = evaluated["aggregate"]["semantic"]["1"]
    assert metrics == {"recall": 0.0, "precision": 0.0, "mrr": 0.0}
    assert aggregate["unanswerable_case_count"] == 1
    assert aggregate["answerable_only_macro"] == {"recall": 0.0, "precision": 0.0, "mrr": 0.0}
