import pytest

from app.evaluation.metrics import calculate_metrics, precision_at_k, recall_at_k, reciprocal_rank_at_k


def test_metrics_use_unique_results_and_fixed_k_denominators():
    ranked = ["relevant-a", "relevant-a", "other", "relevant-b"]
    relevant = {"relevant-a", "relevant-b"}
    assert recall_at_k(ranked, relevant, 2) == pytest.approx(0.5)
    assert precision_at_k(ranked, relevant, 2) == pytest.approx(0.5)
    assert reciprocal_rank_at_k(ranked, relevant, 2) == pytest.approx(1.0)
    assert calculate_metrics(ranked, relevant, 4) == {
        "recall": 1.0, "precision": 0.5, "mrr": 1.0
    }


def test_empty_rankings_and_unanswerable_queries_receive_zero():
    assert calculate_metrics([], {"known"}, 5) == {"recall": 0.0, "precision": 0.0, "mrr": 0.0}
    assert calculate_metrics(["unrelated"], set(), 5) == {"recall": 0.0, "precision": 0.0, "mrr": 0.0}


@pytest.mark.parametrize("k", [0, -1, True, 1.5])
def test_metrics_reject_invalid_k(k):
    with pytest.raises(ValueError):
        recall_at_k([], set(), k)


def test_metrics_reject_empty_chunk_identifiers():
    with pytest.raises(ValueError):
        precision_at_k([""], set(), 1)
    with pytest.raises(ValueError):
        reciprocal_rank_at_k([], {"  "}, 1)
