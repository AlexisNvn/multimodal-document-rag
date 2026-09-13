import math

import pytest

from rag.eval.generation import token_f1
from rag.eval.retrieval import retrieval_metrics

pytestmark = pytest.mark.eval


def test_hand_calculated_graded_ranking():
    result = retrieval_metrics(["b", "x", "a"], {"a": 2, "b": 1}, k=3)
    assert result["recall@3"] == 1
    assert result["precision@3"] == pytest.approx(2 / 3)
    assert result["mrr@3"] == 1
    assert result["ndcg@3"] == pytest.approx((1 + 3 / 2) / (3 + 1 / math.log2(3)))


def test_duplicates_empty_and_missing_relevant_pages():
    assert retrieval_metrics(["a", "a"], {"a": 1, "b": 1}, 2)["recall@2"] == 0.5
    assert retrieval_metrics([], {"a": 2})["ndcg@5"] == 0
    assert retrieval_metrics(["x"], {})["mrr@5"] == 0
    with pytest.raises(ValueError):
        retrieval_metrics([], {}, 0)


def test_f1_counts_repetitions():
    assert token_f1("a a b", "a b") == pytest.approx(0.8)
    assert token_f1("", "") == 1
    assert token_f1("no", "yes") == 0
