# Evaluation

Run `uv run pytest tests/unit`, `tests/integration`, and `tests/eval` independently. The suite
is offline and deterministic. Controlled adapter tests do not establish model quality or GPU
compatibility.

Run `scripts/benchmark.py DATASET_FOLDER` after ingesting the same folder into an isolated
index. It refuses a different or mixed corpus. Reports contain per-query rankings, retrieval
latency, dataset revision, model choices, filters, and aggregates. Use `--mode dense`, `sparse`,
`visual`, or `hybrid` for ablations. `--language` uses the exact language value in downloaded
queries. `--generation` also invokes the answer backend; remote providers may charge for calls.

Metrics use page IDs and graded qrels: recall@k, precision@k, reciprocal rank@k, and nDCG@k
with gain `2^relevance - 1`. IDs are deduplicated before scoring. Missing relevant pages stay
in recall's denominator and ideal DCG. Unjudged queries are excluded from benchmark averages
and counted as skipped; metric functions return zero on empty relevance sets. Translations
remain separate unless a language filter is supplied.

Partial corpora are flagged and report relevant-page coverage. They do not approximate full
quality: fewer distractors and missing positives both change the problem. Query limits are
recorded too. Hash retrieval and extractive answers are engineering baselines, not semantic
model evaluation. Consult the [official evaluation project](https://github.com/illuin-tech/vidore-benchmark)
for leaderboard submission rules.

Generation reports token F1 against references and citation-label validity. These measure
lexical overlap and reference integrity, not correctness, entailment, or citation coverage
of every claim. Add human review of answer correctness, claim support and visual understanding.

`evals/golden_queries.jsonl` is a synthetic regression fixture, separate from ViDoRe. The
analysis notebook reads generated reports; it does not include fabricated experiment results.
