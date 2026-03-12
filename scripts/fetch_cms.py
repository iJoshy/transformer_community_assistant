#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from ingestion import FirebaseCMSExtractor, write_records_json, write_records_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch community records from Firebase-backed CMS and export a normalized snapshot."
    )
    parser.add_argument(
        "--collection",
        default="projects",
        help="Firestore collection to read from.",
    )
    parser.add_argument(
        "--output",
        default="data/community.jsonl",
        help="Path to the normalized export file.",
    )
    parser.add_argument(
        "--format",
        default="jsonl",
        choices=["jsonl", "json"],
        help="Export format.",
    )
    parser.add_argument(
        "--project-types",
        default="",
        help="Optional comma-separated projectType filter.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional max number of normalized records to export.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_types = [
        item.strip() for item in args.project_types.split(",") if item.strip()
    ]

    extractor = FirebaseCMSExtractor(collection_name=args.collection)
    records = extractor.fetch_normalized_records(project_types=project_types or None)
    if args.limit > 0:
        records = records[: args.limit]

    if args.format == "json":
        output = write_records_json(records, args.output)
    else:
        output = write_records_jsonl(records, args.output)

    print(f"Collection: {args.collection}")
    print(f"Records exported: {len(records)}")
    print(f"Format: {args.format}")
    print(f"Output: {output}")


if __name__ == "__main__":
    main()
