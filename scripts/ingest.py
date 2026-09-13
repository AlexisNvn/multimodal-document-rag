import argparse
import json
from collections import defaultdict
from pathlib import Path

from rag.config import Settings
from rag.eval.datasets import load_corpus
from rag.service import RAGService


def main():
    parser = argparse.ArgumentParser(description="Ingest a PDF or a downloaded ViDoRe folder")
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    service = RAGService(Settings())
    if args.path.is_dir():
        _, pages = load_corpus(args.path)
        grouped = defaultdict(list)
        for page in pages:
            grouped[page.document_id].append(page)
        for document_id, group in grouped.items():
            print(json.dumps(service.pipeline.ingest_pages(document_id, document_id, group)))
    else:
        print(json.dumps(service.pipeline.ingest_pdf(args.path)))


if __name__ == "__main__":
    main()
