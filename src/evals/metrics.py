from __future__ import annotations

from typing import Any, Iterable, Sequence

from assistant import AssistantResponse

from .contracts import OfflineEvalCase, OfflineEvalResult


def normalize_text(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def contains_all(text: str, phrases: Sequence[str]) -> bool:
    haystack = normalize_text(text)
    return all(normalize_text(phrase) in haystack for phrase in phrases)


def contains_none(text: str, phrases: Sequence[str]) -> bool:
    haystack = normalize_text(text)
    return all(normalize_text(phrase) not in haystack for phrase in phrases)


def evaluate_retrieval(case: OfflineEvalCase, response: AssistantResponse) -> bool:
    if not case.expected_source_ids:
        return True
    actual = set(response.retrieved_source_ids)
    return all(source_id in actual for source_id in case.expected_source_ids)


def evaluate_answer_relevance(case: OfflineEvalCase, response: AssistantResponse) -> bool:
    if not case.expected_answer_points:
        return True
    return contains_all(response.answer, case.expected_answer_points)


def evaluate_faithfulness(case: OfflineEvalCase, response: AssistantResponse) -> bool:
    if not contains_none(response.answer, case.forbidden_answer_points):
        return False

    if not case.expected_answer_points or not response.retrieved_context.strip():
        return True

    return contains_all(response.retrieved_context, case.expected_answer_points)


def evaluate_tool_correctness(case: OfflineEvalCase, response: AssistantResponse) -> bool:
    if not case.expected_tool:
        return not response.tool_called

    matching_record = next(
        (record for record in response.tool_calls if record.name == case.expected_tool),
        None,
    )
    if matching_record is None:
        return False
    for key, expected_value in case.expected_tool_args.items():
        if matching_record.args.get(key) != expected_value:
            return False
    return matching_record.status == "success"


def evaluate_case(case: OfflineEvalCase, response: AssistantResponse) -> OfflineEvalResult:
    retrieval_relevant = evaluate_retrieval(case, response)
    answer_relevant = evaluate_answer_relevance(case, response)
    faithful = evaluate_faithfulness(case, response)
    tool_correct = evaluate_tool_correctness(case, response)
    end_to_end_success = retrieval_relevant and answer_relevant and faithful and tool_correct
    return OfflineEvalResult(
        case_id=case.id,
        category=case.category,
        query=case.user_query,
        retrieval_relevant=retrieval_relevant,
        answer_relevant=answer_relevant,
        faithful=faithful,
        tool_correct=tool_correct,
        end_to_end_success=end_to_end_success,
        actual_source_ids=tuple(response.retrieved_source_ids),
        actual_tool_names=tuple(record.name for record in response.tool_calls),
        latency_ms=response.latency_ms,
        answer=response.answer,
    )


def summarize_offline_results(results: Iterable[OfflineEvalResult]) -> dict[str, Any]:
    rows = list(results)
    total = len(rows)
    if total == 0:
        return {
            "total_cases": 0,
            "retrieval_relevance_rate": 0.0,
            "answer_relevance_rate": 0.0,
            "faithfulness_rate": 0.0,
            "tool_correctness_rate": 0.0,
            "end_to_end_success_rate": 0.0,
            "average_latency_ms": 0.0,
        }

    return {
        "total_cases": total,
        "retrieval_relevance_rate": ratio(sum(item.retrieval_relevant for item in rows), total),
        "answer_relevance_rate": ratio(sum(item.answer_relevant for item in rows), total),
        "faithfulness_rate": ratio(sum(item.faithful for item in rows), total),
        "tool_correctness_rate": ratio(sum(item.tool_correct for item in rows), total),
        "end_to_end_success_rate": ratio(sum(item.end_to_end_success for item in rows), total),
        "average_latency_ms": round(sum(item.latency_ms for item in rows) / total, 2),
    }


def summarize_online_events(
    response_events: Sequence[dict[str, Any]],
    feedback_events: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    total_responses = len(response_events)
    total_feedback = len(feedback_events)
    liked_count = sum(event.get("liked") is True or event.get("liked") == "Like" for event in feedback_events)
    disliked_count = sum(event.get("liked") is False or event.get("liked") == "Dislike" for event in feedback_events)
    tool_called_count = 0
    tool_success_count = 0
    retrieval_count = 0
    total_latency = 0

    for event in response_events:
        payload = dict(event.get("response_payload") or {})
        total_latency += int(payload.get("latency_ms") or 0)
        if payload.get("retrieval_used"):
            retrieval_count += 1
        tool_calls = list(payload.get("tool_calls") or [])
        if tool_calls:
            tool_called_count += 1
            if all(record.get("status") == "success" for record in tool_calls):
                tool_success_count += 1

    return {
        "total_responses": total_responses,
        "total_feedback": total_feedback,
        "feedback_coverage_rate": ratio(total_feedback, total_responses),
        "likes": liked_count,
        "dislikes": disliked_count,
        "approval_rate": ratio(liked_count, total_feedback),
        "disapproval_rate": ratio(disliked_count, total_feedback),
        "retrieval_usage_rate": ratio(retrieval_count, total_responses),
        "tool_call_rate": ratio(tool_called_count, total_responses),
        "tool_success_rate": ratio(tool_success_count, tool_called_count),
        "average_latency_ms": round(total_latency / total_responses, 2) if total_responses else 0.0,
    }


def ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return round(numerator / denominator, 4)
