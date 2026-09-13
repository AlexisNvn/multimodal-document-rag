# Ingestion

`scripts/ingest.py file.pdf` and `POST /documents` share the pipeline. SHA-256 of bytes produces
a stable document ID. PDFium creates 1.5x PNG renders and extracts embedded text. The default
does not OCR scanned pages; use Docling for those. Docling exports Markdown per page plus
table and figure metadata. API citation page numbers are one-based.

Each page splits at Markdown headings. Child windows use 180 words with 30-word overlap by
default. Tables stay in parent Markdown even when a child crosses rows. This is a word bound,
not a model-token guarantee; long parents are truncated by the generation character budget.

Embeddings are computed before the database transaction. Reingestion replaces the document
atomically. Rendered artifacts are outside the transaction and may remain after failures or
deletion; they are not retrieved without database records. Do not run simultaneous independent
ingestion CLI processes against the same artifact directory.

The [ViDoRe dataset](https://huggingface.co/datasets/vidore/vidore_v3_computer_science) has four
configs: `corpus`, `queries`, `qrels`, and `documents_metadata`, each with a `test` split.
The adapter uses `corpus_id`, `image`, `doc_id`, `markdown`, and `page_number_in_doc`. Page IDs
are namespaced by repository. Raw zero-based page numbers are preserved in metadata; citations
add one. Queries and qrels are downloaded for evaluation but never embedded. Duplicate relevance
judgments use the maximum grade.

The downloader records the commit SHA, source URL, counts, and partial-corpus flag. Use
`--revision SHA` to reproduce a download. It saves the dataset card and document metadata so
publisher licensing survives export. Output directories must be empty to prevent mixing
snapshots; interrupted downloads should be retried into a new folder. Hugging Face caches
shards. Streaming a small sample may still transfer a whole Parquet shard.

On Windows, keep `HF_DATASETS_CACHE` short if the checkout is deeply nested; some dataset
lock filenames repeat the absolute cache path and can exceed Windows' path-length limit.
For a corporate HTTPS proxy, set `SSL_CERT_FILE` and `REQUESTS_CA_BUNDLE` to a trusted CA bundle;
do not disable certificate verification.

The initial 24-page sample is selected without qrels. All queries and qrels are retained.
A full benchmark must index all pages, including distractors, in an isolated index. A ViDoRe
page dataset bypasses PDF parsing: supplied Markdown is the text channel, PNGs the visual channel.
