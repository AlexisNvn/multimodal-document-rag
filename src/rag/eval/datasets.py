"""ViDoRe V3's corpus/queries/qrels adapter. Labels never enter the retrieval index."""

import hashlib
import json
from pathlib import Path

from rag.db.models import Page

DOMAINS = (
    "hr",
    "finance_en",
    "industrial",
    "pharmaceuticals",
    "computer_science",
    "energy",
    "physics",
    "finance_fr",
)


def write_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def page_id(dataset: str, corpus_id) -> str:
    return f"{dataset}:{corpus_id}"


def corpus_page(row, dataset: str, image_path: str | None = None) -> Page:
    return Page(
        page_id(dataset, row["corpus_id"]),
        f"{dataset}:{row['doc_id']}",
        int(row["page_number_in_doc"]) + 1,
        row.get("markdown") or "",
        image_path,
        {
            "dataset": dataset,
            "corpus_id": str(row["corpus_id"]),
            "original_page_number": row["page_number_in_doc"],
        },
    )


def download_vidore(domain: str, output: Path, limit: int | None = None, revision="main"):
    from datasets import load_dataset
    from huggingface_hub import HfApi, hf_hub_download

    if domain not in DOMAINS or (limit is not None and limit < 1):
        raise ValueError("Choose a ViDoRe V3 domain and a positive page limit")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output must be empty to avoid mixing dataset revisions")
    dataset = f"vidore/vidore_v3_{domain}"
    sha = HfApi().dataset_info(dataset, revision=revision).sha
    output.mkdir(parents=True, exist_ok=True)
    (output / "images").mkdir(exist_ok=True)
    card = hf_hub_download(dataset, "README.md", repo_type="dataset", revision=sha)
    (output / "DATASET_CARD.md").write_text(
        Path(card).read_text(encoding="utf-8"), encoding="utf-8"
    )
    records = []
    corpus = load_dataset(dataset, "corpus", split="test", revision=sha, streaming=True)
    from itertools import islice

    selected = islice(corpus, limit) if limit is not None else corpus
    for row in selected:
        image_name = hashlib.sha256(str(row["corpus_id"]).encode()).hexdigest()[:20] + ".png"
        image_path = output / "images" / image_name
        row["image"].convert("RGB").save(image_path)
        records.append(
            {key: value for key, value in row.items() if key != "image"}
            | {"image_path": f"images/{image_name}"}
        )
    write_jsonl(output / "corpus.jsonl", records)
    counts = {"corpus": len(records)}
    for config in ("queries", "qrels", "documents_metadata"):
        rows = list(load_dataset(dataset, config, split="test", revision=sha, streaming=True))
        write_jsonl(output / f"{config}.jsonl", rows)
        counts[config] = len(rows)
    manifest = {
        "dataset": dataset,
        "revision": sha,
        "split": "test",
        "counts": counts,
        "partial_corpus": limit is not None,
        "page_limit": limit,
        "source": f"https://huggingface.co/datasets/{dataset}",
        "annotation_license": "CC-BY-4.0; source documents retain publisher licenses",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def load_corpus(folder: Path):
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    root = folder.resolve()
    pages = []
    for row in read_jsonl(folder / "corpus.jsonl"):
        image = (root / row["image_path"]).resolve()
        if not image.is_relative_to(root) or not image.is_file():
            raise ValueError("Corpus image must exist inside the dataset folder")
        page = corpus_page(row, manifest["dataset"], str(image))
        page.metadata["dataset_revision"] = manifest["revision"]
        pages.append(page)
    return manifest, pages


def load_queries(folder: Path):
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    qrels = {}
    for row in read_jsonl(folder / "qrels.jsonl"):
        labels = qrels.setdefault(str(row["query_id"]), {})
        key = page_id(manifest["dataset"], row["corpus_id"])
        labels[key] = max(labels.get(key, 0), float(row["score"]))
    return read_jsonl(folder / "queries.jsonl"), qrels
