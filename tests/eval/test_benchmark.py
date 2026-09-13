import json

import pytest
from PIL import Image

from rag.eval.benchmark import benchmark
from rag.eval.datasets import load_corpus, write_jsonl

pytestmark = pytest.mark.eval


@pytest.fixture
def corpus_folder(tmp_path, service):
    folder = tmp_path / "dataset"
    folder.mkdir()
    Image.new("RGB", (8, 8)).save(folder / "page.png")
    (folder / "manifest.json").write_text(
        json.dumps(
            {
                "dataset": "vidore/test",
                "revision": "abc123",
                "partial_corpus": True,
            }
        )
    )
    write_jsonl(
        folder / "corpus.jsonl",
        [
            {
                "corpus_id": 0,
                "doc_id": "book",
                "page_number_in_doc": 0,
                "markdown": "Revenue increased",
                "image_path": "page.png",
            }
        ],
    )
    write_jsonl(
        folder / "queries.jsonl",
        [
            {"query_id": 1, "query": "Revenue", "language": "en", "answer": "Revenue increased"},
            {"query_id": 2, "query": "Unjudged", "language": "en"},
        ],
    )
    write_jsonl(
        folder / "qrels.jsonl",
        [
            {"query_id": 1, "corpus_id": 0, "score": 2},
            {"query_id": 1, "corpus_id": 99, "score": 1},
        ],
    )
    _, pages = load_corpus(folder)
    service.pipeline.ingest_pages(pages[0].document_id, "book", pages)
    return folder


def test_partial_benchmark_retains_missing_positive_denominator(service, corpus_folder):
    result = benchmark(service, corpus_folder, k=1, mode="sparse", generation=True)
    assert result["metrics"]["recall@1"] == 0.5
    assert result["relevant_page_coverage"] == 0.5
    assert result["dataset"]["partial_corpus"]
    assert result["evaluated_queries"] == 1
    assert result["skipped_unjudged_queries"] == 1
    assert result["queries"][0]["generation"]["citation_labels_valid"]


def test_benchmark_rejects_mixed_corpus(service, corpus_folder, pages):
    service.pipeline.ingest_pages("doc", "Other", pages)
    with pytest.raises(ValueError, match="exactly"):
        benchmark(service, corpus_folder)


def test_benchmark_rejects_changed_revision(service, corpus_folder):
    path = corpus_folder / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["revision"] = "different"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="revision"):
        benchmark(service, corpus_folder)


def test_empty_language_filter_fails(service, corpus_folder):
    with pytest.raises(ValueError, match="No queries"):
        benchmark(service, corpus_folder, language="missing")
