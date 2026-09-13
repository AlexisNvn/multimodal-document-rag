.PHONY: install serve test test-unit test-integration test-eval lint download ingest benchmark
install:
	uv sync --frozen
serve:
	uv run uvicorn rag.api.app:app --reload
test:
	uv run pytest
test-unit:
	uv run pytest tests/unit
test-integration:
	uv run pytest tests/integration
test-eval:
	uv run pytest tests/eval
lint:
	uv run ruff check .
	uv run ruff format --check .
download:
	uv run python scripts/download_vidore.py --limit 24
ingest:
	uv run python scripts/ingest.py data/vidore/computer_science
benchmark:
	uv run python scripts/benchmark.py data/vidore/computer_science --mode sparse --limit 20
