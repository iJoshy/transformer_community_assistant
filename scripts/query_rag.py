#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from rag import RagConfig, RagRetriever
from runtime_env import ensure_dotenv_loaded


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query a Chroma vector store.")
    parser.add_argument("--query", required=True, help="Question or search query.")
    parser.add_argument("--persist-dir", default="vector_db", help="Chroma persistence directory.")
    parser.add_argument(
        "--embedding-model",
        default=None,
        help="Embedding model used to load the vector store. Defaults by provider env.",
    )
    parser.add_argument("--k", type=int, default=4, help="Number of chunks to retrieve.")
    parser.add_argument("--max-chars", type=int, default=4000, help="Max characters in context.")
    parser.add_argument("--json", action="store_true", help="Output JSON with docs and context.")
    return parser.parse_args()


def main() -> None:
    ensure_dotenv_loaded()
    args = parse_args()
    config = RagConfig(
        persist_dir=args.persist_dir,
        embedding_model=args.embedding_model,
    )

    retriever = RagRetriever.from_config(config=config)
    result = retriever.retrieve(
        args.query,
        k=args.k,
        max_chars=args.max_chars,
    )

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
        return

    if result.source_ids:
        print("Source IDs:")
        for source_id in result.source_ids:
            print(f"- {source_id}")
        print()

    print("Context:\n")
    print(result.context)


if __name__ == "__main__":
    main()
