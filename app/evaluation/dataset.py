"""Load and validate the reviewable JSON evaluation dataset."""
import json
from pathlib import Path
from typing import Any

CATEGORIES = {"exact_identifier", "technical_terminology", "paraphrase", "ordinary_question", "unanswerable"}

class DatasetValidationError(ValueError):
    """Raised when the evaluation file has invalid schema or relevance labels."""


def load_dataset(path: Path, available_sources: set[str]) -> dict[str, Any]:
    """Load cases and ensure every relevance label points to a real corpus source."""
    try:
        dataset = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetValidationError(f"Could not read evaluation dataset: {exc}") from exc
    if not isinstance(dataset, dict) or type(dataset.get("version")) is not int or dataset.get("version") != 1:
        raise DatasetValidationError("Dataset must be an object with version 1.")
    cases = dataset.get("cases")
    if not isinstance(cases, list) or not cases:
        raise DatasetValidationError("Dataset cases must be a non-empty list.")

    seen_case_ids = set()
    for index, case in enumerate(cases):
        prefix = f"cases[{index}]"
        if not isinstance(case, dict):
            raise DatasetValidationError(f"{prefix} must be an object.")
        case_id = case.get("case_id")
        if not isinstance(case_id, str) or not case_id.strip() or case_id in seen_case_ids:
            raise DatasetValidationError(f"{prefix}.case_id must be a unique non-empty string.")
        seen_case_ids.add(case_id)
        if not isinstance(case.get("question"), str) or not case["question"].strip():
            raise DatasetValidationError(f"{case_id}.question must be non-empty.")
        if not isinstance(case.get("category"), str) or case.get("category") not in CATEGORIES:
            raise DatasetValidationError(f"{case_id}.category is not supported.")
        sources = case.get("relevant_sources")
        if not isinstance(sources, list) or any(not isinstance(source, str) for source in sources):
            raise DatasetValidationError(f"{case_id}.relevant_sources must be a list of source IDs.")
        if len(sources) != len(set(sources)):
            raise DatasetValidationError(f"{case_id}.relevant_sources contains duplicates.")
        unknown = set(sources) - available_sources
        if unknown:
            raise DatasetValidationError(f"{case_id} references unknown sources: {sorted(unknown)}")
        is_unanswerable = case["category"] == "unanswerable"
        if is_unanswerable != (len(sources) == 0):
            raise DatasetValidationError(
                f"{case_id}: unanswerable cases must have no relevant sources; other cases need at least one."
            )
        if not isinstance(case.get("relevance_note"), str) or not case["relevance_note"].strip():
            raise DatasetValidationError(f"{case_id}.relevance_note must explain the label.")
    return dataset
