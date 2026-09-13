"""Benchmark runner and CLI, importable from the installed rag package."""

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from rag.config import Settings
from rag.eval.datasets import load_corpus, load_queries
from rag.eval.generation import generation_metrics
from rag.eval.retrieval import retrieval_metrics
from rag.service import RAGService


def benchmark(service, folder, k=5, mode="hybrid", language=None, limit=None, generation=False):
    manifest, corpus = load_corpus(folder)
    queries, qrels = load_queries(folder)
    expected = {page.id for page in corpus}
    stored_pages = service.repository.pages()
    indexed = set(stored_pages)
    if expected != indexed:
        raise ValueError(
            "Benchmark index must contain exactly this downloaded corpus; use a new data dir"
        )
    if any(
        stored_pages[page.id].metadata.get("dataset_revision") != manifest["revision"]
        or stored_pages[page.id].text != page.text
        for page in corpus
    ):
        raise ValueError("Indexed corpus revision/content differs from the benchmark dataset")
    selected = [row for row in queries if language is None or row.get("language") == language]
    if limit is not None:
        if limit < 1:
            raise ValueError("Query limit must be positive")
        selected = selected[:limit]
    rows = []
    for query in selected:
        query_id = str(query["query_id"])
        labels = qrels.get(query_id, {})
        if not any(grade > 0 for grade in labels.values()):
            continue
        start = time.perf_counter()
        hits = service.retrieve(query["query"], k, mode)
        elapsed = time.perf_counter() - start
        row = {
            "query_id": query_id,
            "latency_seconds": elapsed,
            "ranked_ids": [hit.page.id for hit in hits],
            **retrieval_metrics([hit.page.id for hit in hits], labels, k),
        }
        if generation:
            answer, _ = service.answer(query["query"], k, mode)
            references = query.get("raw_answers") or [query.get("answer", "")]
            row["generation"] = generation_metrics(answer, references)
        rows.append(row)
    if not rows:
        raise ValueError("No queries with positive relevance judgments match the selection")
    metric_names = [f"{metric}@{k}" for metric in ("recall", "precision", "mrr", "ndcg")]
    return {
        "created_at": datetime.now(UTC).isoformat(),
        "dataset": manifest,
        "mode": mode,
        "k": k,
        "language": language,
        "query_limit": limit,
        "text_backend": service.settings.text_backend,
        "text_model": service.settings.text_model,
        "visual_backend": service.settings.visual_backend,
        "visual_model": service.settings.visual_model,
        "reranker": service.settings.reranker_model,
        "answer_backend": service.settings.answer_backend,
        "evaluated_queries": len(rows),
        "skipped_unjudged_queries": len(selected) - len(rows),
        "relevant_page_coverage": len(
            {p for labels in qrels.values() for p, g in labels.items() if g > 0} & expected
        )
        / max(1, len({p for labels in qrels.values() for p, g in labels.items() if g > 0})),
        "metrics": {
            metric: float(np.mean([row[metric] for row in rows])) for metric in metric_names
        },
        "latency_p95_seconds": float(np.percentile([r["latency_seconds"] for r in rows], 95)),
        "queries": rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path, default=Path("evals/results/benchmark.json"))
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--mode", choices=["dense", "sparse", "visual", "hybrid"], default="hybrid")
    parser.add_argument("--language")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--generation", action="store_true")
    args = parser.parse_args()
    result = benchmark(
        RAGService(Settings()),
        args.dataset,
        args.k,
        args.mode,
        args.language,
        args.limit,
        args.generation,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "queries"}, indent=2))


if __name__ == "__main__":
    main()
