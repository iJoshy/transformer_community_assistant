from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Sequence

from assistant import AssistantResponse

from .contracts import OfflineEvalCase
from .metrics import evaluate_case, summarize_offline_results
from .storage import load_jsonl, write_jsonl


DEFAULT_OFFLINE_CASES_PATH = Path(__file__).with_name("offline_cases.jsonl")
DEFAULT_OFFLINE_RESULTS_PATH = Path("data/evals/offline_eval_results.jsonl")


def load_offline_cases(path: str | Path = DEFAULT_OFFLINE_CASES_PATH) -> list[OfflineEvalCase]:
    rows = load_jsonl(path)
    return [OfflineEvalCase.from_dict(row) for row in rows]


def run_offline_evals(
    assistant_service: Any,
    *,
    cases: Sequence[OfflineEvalCase] | None = None,
    cases_path: str | Path = DEFAULT_OFFLINE_CASES_PATH,
    output_path: str | Path | None = DEFAULT_OFFLINE_RESULTS_PATH,
    k: int = 4,
    max_chars: int = 4000,
) -> dict[str, Any]:
    eval_cases = list(cases or load_offline_cases(cases_path))
    results = []

    for case in eval_cases:
        response: AssistantResponse = assistant_service.answer(
            case.user_query,
            history=[],
            k=k,
            max_chars=max_chars,
        )
        results.append(evaluate_case(case, response))

    if output_path is not None:
        write_jsonl(output_path, [result.to_dict() for result in results])

    summary = summarize_offline_results(results)
    return {
        "summary": summary,
        "results": [result.to_dict() for result in results],
    }
