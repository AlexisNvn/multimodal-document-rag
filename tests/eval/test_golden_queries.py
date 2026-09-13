from pathlib import Path

import pytest

from rag.eval.datasets import read_jsonl
from rag.eval.retrieval import retrieval_metrics

pytestmark = pytest.mark.eval


def test_synthetic_golden_retrieval(indexed):
    path = Path(__file__).resolve().parents[2] / "evals" / "golden_queries.jsonl"
    for row in read_jsonl(path):
        hits = indexed.retrieve(row["query"], k=1, mode="sparse")
        assert retrieval_metrics([hit.page.id for hit in hits], row["qrels"], 1)["recall@1"] == 1
