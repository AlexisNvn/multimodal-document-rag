# Multimodal Document RAG

A portfolio implementation of page-grounded document retrieval and answering. Ingest PDFs or
ViDoRe V3 page images, retrieve with BM25, dense text embeddings and optional ColQwen2 visual
embeddings, fuse rankings, and return answers with document/page citations.

The default configuration runs on a CPU without API keys or model downloads. It uses a
**lexical hashing smoke-test encoder**, BM25, and **extractive source excerpts**. Enable the
model extras below for semantic retrieval, OCR/layout analysis, visual retrieval, and synthesis.
The default is labeled in API health responses and benchmark reports.

## Quick start

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and Python 3.12, then:

```sh
uv sync --frozen
cp .env.example .env
uv run uvicorn rag.api.app:app --reload
```

On PowerShell, use `Copy-Item .env.example .env`. Open http://localhost:8000/docs for the
interactive API, upload a PDF at `POST /documents`, then call `POST /query`:

```json
{"query": "What was annual revenue?", "k": 5, "mode": "hybrid"}
```

`POST /search` returns ranked pages without generation. `GET /documents` lists the index;
`DELETE /documents/{document_id}` removes a document and its indexed vectors.
Content-hashed PDF IDs make repeated ingestion idempotent. Rendered files remain on disk
after deletion; the API stops retrieving them. The API is for local use and has no authentication.

Alternatively run `docker compose up --build`. The container uses offline defaults,
binds to localhost, and persists SQLite plus images in the `rag-data` volume.

## ViDoRe V3

The downloader supports all eight domains in the
[official ViDoRe V3 collection](https://huggingface.co/collections/vidore/vidore-benchmark-v3).
It resolves a commit SHA, exports page images and Markdown, preserves query/relevance tables
and document licensing metadata, and writes a manifest. Network access is required.

```sh
# Small initial run. Streaming can still fetch a large Parquet shard.
uv run python scripts/download_vidore.py --domain computer_science --limit 24
uv run python scripts/ingest.py data/vidore/computer_science
uv run python scripts/benchmark.py data/vidore/computer_science --mode sparse --limit 20
```

Omit `--limit` when downloading to an **empty, separate output folder** for the full corpus:

```sh
uv run python scripts/download_vidore.py --domain computer_science --output data/vidore/cs-full
```

Use a new `RAG_DATA_DIR` for the full index, then ingest that folder and benchmark it.
Partial downloads retain all qrels and are marked `partial_corpus: true`; missing relevant
pages continue to count against recall. These are smoke checks, not leaderboard scores.
The computer science corpus is roughly 0.5 GB upstream, before exported images and caches.
See the [dataset card](https://huggingface.co/datasets/vidore/vidore_v3_computer_science)
for schema, attribution, and document-specific licenses. `evals/golden_queries.jsonl` contains
only labeled synthetic regression fixtures, not ViDoRe annotations.

## Enable models

```sh
uv sync --frozen --extra models       # semantic text embeddings and cross-encoder reranking
uv sync --frozen --extra docling      # OCR, layout, figures and tables
uv sync --frozen --extra visual      # ColQwen2 page-image retrieval (large model)
# Combine multiple --extra flags to retain all selected extras.
```

Set the following in `.env`, then ingest into a **new** `RAG_DATA_DIR`:

```dotenv
RAG_DATA_DIR=data/semantic-index
RAG_TEXT_BACKEND=sentence-transformers
RAG_TEXT_MODEL=sentence-transformers/all-MiniLM-L6-v2
RAG_PARSER=docling
RAG_VISUAL_BACKEND=colqwen2
RAG_VISUAL_MODEL=vidore/colqwen2-v1.0
RAG_DEVICE=cuda:0
# Optional, text-only reranker:
RAG_RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
```

These are example model choices, not claims of benchmark-leading performance. MiniLM is an
English baseline; choose a multilingual model for multilingual experiments. Weights download
on first use. ColQwen2 needs substantial RAM/VRAM and is slower on CPU. GPU installation depends
on your CUDA/PyTorch environment. Heavy model execution is separate from the offline test suite.
Index configuration mismatches fail explicitly.

For synthesized multimodal answers, configure a chat-completions compatible **vision** endpoint:

```dotenv
RAG_ANSWER_BACKEND=chat
RAG_LLM_BASE_URL=http://localhost:8001/v1
RAG_LLM_MODEL=your-vision-model
RAG_LLM_API_KEY=
```

This sends retrieved text and up to four page images to that endpoint. Invalid or missing
citation labels trigger abstention. Label validation is not proof of factual grounding.
Without an endpoint, extractive generation returns excerpts and abstains for image-only evidence.

## Separate tests

```sh
uv run pytest tests/unit
uv run pytest tests/integration
uv run pytest tests/eval
uv run pytest --cov=rag --cov-report=term-missing
uv run ruff check .
uv run ruff format --check .
```

Unit tests cover ingestion, chunking, embeddings, retrieval, generation and dataset adapters.
Integration tests exercise real PDF rendering, HTTP upload/query/delete, restart persistence,
transactions and configuration compatibility. Evaluation tests use hand-calculated metrics
and synthetic golden queries. Tests need no API credentials, remote services, or ML weights.

See [architecture](docs/architecture.md), [ingestion](docs/ingestion.md), and
[evaluation](docs/evaluation.md). `notebooks/retrieval_analysis.ipynb` explores saved reports.
This version uses exact vector scans and synchronous ingestion; larger deployments need
an indexed vector store, background workers and authenticated access.

## Verification in this checkout

50 offline tests pass (86% coverage); Ruff lint/format checks and the wheel/sdist build pass.
A real 24-page computer-science sample is available locally in `data/vidore/cs-smoke`, with
1,290 queries and 6,294 qrels. It is indexed in `data/rag.sqlite3`. The saved
[smoke report](evals/results/vidore-v3-smoke.json) evaluates all 1,290 queries against that
partial corpus using BM25. Only about 2% of relevant pages are present; the report verifies
execution, not retrieval quality. Data and caches are gitignored.

Large model inference and a live answer provider have not been exercised. Docker Compose
configuration validates, but container execution was not tested because the daemon was stopped.
