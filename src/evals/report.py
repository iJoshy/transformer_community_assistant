from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .metrics import summarize_offline_results, summarize_online_events
from .offline import DEFAULT_OFFLINE_RESULTS_PATH
from .storage import load_jsonl
from .online_feedback import DEFAULT_FEEDBACK_LOG_PATH, DEFAULT_RESPONSE_LOG_PATH


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize offline and online eval signals.")
    parser.add_argument(
        "--offline-results",
        default=str(DEFAULT_OFFLINE_RESULTS_PATH),
        help="Path to offline eval result JSONL.",
    )
    parser.add_argument(
        "--responses",
        default=str(DEFAULT_RESPONSE_LOG_PATH),
        help="Path to online response event JSONL.",
    )
    parser.add_argument(
        "--feedback",
        default=str(DEFAULT_FEEDBACK_LOG_PATH),
        help="Path to online feedback event JSONL.",
    )
    parser.add_argument("--json", action="store_true", help="Print raw JSON output.")
    return parser.parse_args(argv)


def build_report(
    *,
    offline_results_path: str | Path = DEFAULT_OFFLINE_RESULTS_PATH,
    responses_path: str | Path = DEFAULT_RESPONSE_LOG_PATH,
    feedback_path: str | Path = DEFAULT_FEEDBACK_LOG_PATH,
) -> dict:
    offline_rows = load_jsonl(offline_results_path)
    response_rows = load_jsonl(responses_path)
    feedback_rows = load_jsonl(feedback_path)

    offline_summary = summarize_offline_results(_rows_to_result_objects(offline_rows))
    online_summary = summarize_online_events(response_rows, feedback_rows)
    return {
        "offline_summary": offline_summary,
        "online_summary": online_summary,
    }


def print_report(report: dict, *, as_json: bool = False) -> None:
    if as_json:
        print(json.dumps(report, indent=2))
        return

    print("Offline summary:")
    for key, value in report["offline_summary"].items():
        print(f"- {key}: {value}")

    print("\nOnline summary:")
    for key, value in report["online_summary"].items():
        print(f"- {key}: {value}")


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    report = build_report(
        offline_results_path=args.offline_results,
        responses_path=args.responses,
        feedback_path=args.feedback,
    )
    print_report(report, as_json=args.json)


def _rows_to_result_objects(rows: list[dict]) -> list:
    from .contracts import OfflineEvalResult

    return [
        OfflineEvalResult(
            case_id=str(row["case_id"]),
            category=str(row["category"]),
            query=str(row["query"]),
            retrieval_relevant=bool(row["retrieval_relevant"]),
            answer_relevant=bool(row["answer_relevant"]),
            faithful=bool(row["faithful"]),
            tool_correct=bool(row["tool_correct"]),
            end_to_end_success=bool(row["end_to_end_success"]),
            actual_source_ids=tuple(row.get("actual_source_ids") or ()),
            actual_tool_names=tuple(row.get("actual_tool_names") or ()),
            latency_ms=int(row.get("latency_ms") or 0),
            answer=str(row.get("answer") or ""),
        )
        for row in rows
    ]


if __name__ == "__main__":
    main()
