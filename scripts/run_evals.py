#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SRC))

import argparse
import json

from assistant import CommunityAssistantService
from evals.offline import run_offline_evals
from model_provider import get_default_chat_model, get_default_embedding_model
from rag import RagConfig
from runtime_env import ensure_dotenv_loaded


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run offline assistant eval cases.")
    parser.add_argument(
        "--cases",
        default=str(SRC / "evals" / "offline_cases.jsonl"),
        help="Path to the offline eval case JSONL.",
    )
    parser.add_argument(
        "--output",
        default="data/evals/offline_eval_results.jsonl",
        help="Path to the offline eval result JSONL.",
    )
    parser.add_argument("--persist-dir", default="vector_db", help="Chroma persistence directory.")
    parser.add_argument(
        "--embedding-model",
        default=None,
        help="Embedding model used to load the vector store. Defaults by provider env.",
    )
    parser.add_argument(
        "--chat-model",
        default=None,
        help="Chat model used by the assistant. Defaults by provider env.",
    )
    parser.add_argument("--temperature", type=float, default=0.0, help="Chat model temperature.")
    parser.add_argument("--k", type=int, default=4, help="Number of chunks to retrieve.")
    parser.add_argument("--max-chars", type=int, default=4000, help="Maximum retrieved context length.")
    parser.add_argument("--json", action="store_true", help="Print JSON output.")
    return parser.parse_args()


def main() -> None:
    ensure_dotenv_loaded()
    args = parse_args()
    rag_config = RagConfig(
        persist_dir=args.persist_dir,
        embedding_model=args.embedding_model or get_default_embedding_model(),
    )
    assistant_service = CommunityAssistantService.from_env(
        rag_config=rag_config,
        model=args.chat_model or get_default_chat_model(),
        temperature=args.temperature,
    )
    report = run_offline_evals(
        assistant_service,
        cases_path=args.cases,
        output_path=args.output,
        k=args.k,
        max_chars=args.max_chars,
    )
    if args.json:
        print(json.dumps(report, indent=2))
        return

    print("Offline eval summary:")
    for key, value in report["summary"].items():
        print(f"- {key}: {value}")
    print(f"\nDetailed results written to: {args.output}")


if __name__ == "__main__":
    main()
