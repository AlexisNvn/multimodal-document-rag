import json

import pytest

from rag.eval.datasets import corpus_page, load_corpus, load_queries, write_jsonl

pytestmark = pytest.mark.unit


def test_vidore_ids_and_zero_based_page_number():
    row = {"corpus_id": 0, "doc_id": "book", "page_number_in_doc": 0, "markdown": "Text"}
    page = corpus_page(row, "vidore/test")
    assert page.id == "vidore/test:0"
    assert page.number == 1
    assert page.metadata["original_page_number"] == 0


def test_relevance_join_preserves_grades_and_string_ids(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({"dataset": "vidore/test"}))
    write_jsonl(tmp_path / "queries.jsonl", [{"query_id": 0, "query": "why?"}])
    write_jsonl(
        tmp_path / "qrels.jsonl",
        [
            {"query_id": 0, "corpus_id": 7, "score": 1},
            {"query_id": 0, "corpus_id": 7, "score": 2},
        ],
    )
    queries, labels = load_queries(tmp_path)
    assert queries[0]["query_id"] == 0
    assert labels == {"0": {"vidore/test:7": 2}}


def test_dataset_rejects_image_path_traversal(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({"dataset": "vidore/test"}))
    write_jsonl(tmp_path / "corpus.jsonl", [{"image_path": "../outside.png"}])
    with pytest.raises(ValueError, match="inside"):
        load_corpus(tmp_path)
