import json

from pathlib import Path
import pytest

from app.evaluation.dataset import DatasetValidationError, load_dataset


def make_case(case_id="q1", category="ordinary_question", sources=None):
    return {"case_id": case_id, "question": "Which policy applies?", "category": category,
            "relevant_sources": sources if sources is not None else ["policy"],
            "relevance_note": "The policy source contains the labeled evidence."}


def test_dataset_schema_accepts_answerable_and_explicit_unanswerable(tmp_path):
    path = tmp_path / "dataset.json"
    path.write_text(json.dumps({"version": 1, "cases": [
        make_case(), make_case("q2", "unanswerable", [])
    ]}), encoding="utf-8")
    loaded = load_dataset(path, {"policy"})
    assert len(loaded["cases"]) == 2
    assert loaded["cases"][1]["relevant_sources"] == []


@pytest.mark.parametrize("case", [
    make_case(category="unanswerable", sources=["policy"]),
    make_case(category="ordinary_question", sources=[]),
    make_case(category="unknown_category"),
    make_case(sources=["missing-source"]),
    {"case_id": "q1", "question": "?", "category": "ordinary_question",
     "relevant_sources": ["policy"]},
])
def test_dataset_rejects_invalid_labels_or_schema(tmp_path, case):
    path = tmp_path / "dataset.json"
    path.write_text(json.dumps({"version": 1, "cases": [case]}), encoding="utf-8")
    with pytest.raises(DatasetValidationError):
        load_dataset(path, {"policy"})



def test_checked_in_dataset_resolves_only_to_real_corpus_sources():
    root = Path(__file__).resolve().parents[1]
    corpus_dir = root / "evaluation" / "corpus"
    source_ids = {path.stem for path in corpus_dir.glob("*.txt")}
    dataset = load_dataset(root / "evaluation" / "dataset.json", source_ids)
    assert len(dataset["cases"]) == 30
    assert sum(case["category"] == "unanswerable" for case in dataset["cases"]) == 4
def test_dataset_rejects_duplicate_case_ids(tmp_path):
    path = tmp_path / "dataset.json"
    path.write_text(json.dumps({"version": 1, "cases": [make_case(), make_case()]}), encoding="utf-8")
    with pytest.raises(DatasetValidationError):
        load_dataset(path, {"policy"})
