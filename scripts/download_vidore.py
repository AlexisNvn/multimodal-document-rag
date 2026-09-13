import argparse
import json
from pathlib import Path

from rag.eval.datasets import DOMAINS, download_vidore


def main():
    parser = argparse.ArgumentParser(description="Download revision-pinned ViDoRe V3 page data")
    parser.add_argument("--domain", choices=DOMAINS, default="computer_science")
    parser.add_argument("--output", type=Path, default=Path("data/vidore/computer_science"))
    parser.add_argument("--limit", type=int, help="Partial corpus smoke run; omit for full corpus")
    parser.add_argument("--revision", default="main")
    args = parser.parse_args()
    print(
        json.dumps(download_vidore(args.domain, args.output, args.limit, args.revision), indent=2)
    )


if __name__ == "__main__":
    main()
