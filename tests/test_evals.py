from __future__ import annotations

from pathlib import Path

from assistant import AssistantResponse, ToolExecutionRecord
from evals.contracts import OfflineEvalCase
from evals.metrics import evaluate_case, summarize_offline_results, summarize_online_events
from evals.offline import load_offline_cases, run_offline_evals
from evals.online_feedback import OnlineFeedbackLogger
from evals.report import build_report


def make_response() -> AssistantResponse:
    return AssistantResponse(
        query="What events are coming up?",
        answer="Community Meetup is on April 2.",
        retrieved_context="Community Meetup is on April 2.",
        retrieved_source_ids=("event-1",),
        retrieved_chunks=(
            {
                "content": "Community Meetup is on April 2.",
                "metadata": {"source_id": "event-1"},
                "source_id": "event-1",
            },
        ),
        retrieval_used=True,
        tool_called=False,
        tool_calls=(),
        latency_ms=125,
    )


class StubAssistantService:
    def __init__(self, response: AssistantResponse):
        self.response = response
        self.calls: list[str] = []

    def answer(self, question: str, *, history, k: int, max_chars: int) -> AssistantResponse:
        self.calls.append(question)
        return self.response


def test_evaluate_case_scores_retrieval_answer_and_faithfulness():
    case = OfflineEvalCase(
        id="case-1",
        category="retrieval",
        user_query="What events are coming up?",
        expected_source_ids=("event-1",),
        expected_answer_points=("Community Meetup", "April 2"),
        forbidden_answer_points=("December 25",),
    )

    result = evaluate_case(case, make_response())

    assert result.retrieval_relevant is True
    assert result.answer_relevant is True
    assert result.faithful is True
    assert result.tool_correct is True
    assert result.end_to_end_success is True


def test_run_offline_evals_writes_results(tmp_path):
    cases_path = tmp_path / "cases.jsonl"
    output_path = tmp_path / "results.jsonl"
    cases_path.write_text(
        '{"id":"case-1","category":"retrieval","user_query":"What events are coming up?","expected_source_ids":["event-1"],"expected_answer_points":["Community Meetup"],"forbidden_answer_points":[],"expected_tool":"","expected_tool_args":{}}\n',
        encoding="utf-8",
    )
    service = StubAssistantService(make_response())

    report = run_offline_evals(
        service,
        cases_path=cases_path,
        output_path=output_path,
    )

    assert report["summary"]["total_cases"] == 1
    assert output_path.exists()
    assert service.calls == ["What events are coming up?"]


def test_online_feedback_logger_and_report_summary(tmp_path):
    logger = OnlineFeedbackLogger(
        response_log_path=tmp_path / "responses.jsonl",
        feedback_log_path=tmp_path / "feedback.jsonl",
    )
    state = logger.new_session_state()
    response = make_response()
    state = logger.record_response(
        session_state=state,
        user_message="What events are coming up?",
        assistant_message=response.answer,
        assistant_message_index=1,
        response=response,
    )
    _, feedback_event = logger.record_feedback(
        session_state=state,
        history=[
            {"role": "user", "content": "What events are coming up?"},
            {"role": "assistant", "content": response.answer},
        ],
        like_data=type("LikeDataStub", (), {"index": 1, "value": response.answer, "liked": True})(),
    )

    report = build_report(
        offline_results_path=tmp_path / "missing-offline.jsonl",
        responses_path=tmp_path / "responses.jsonl",
        feedback_path=tmp_path / "feedback.jsonl",
    )

    assert feedback_event["liked"] is True
    assert report["online_summary"]["total_responses"] == 1
    assert report["online_summary"]["total_feedback"] == 1
    assert report["online_summary"]["approval_rate"] == 1.0


def test_load_offline_cases_reads_packaged_dataset():
    cases = load_offline_cases(Path("src/evals/offline_cases.jsonl"))

    assert len(cases) >= 1
    assert isinstance(cases[0], OfflineEvalCase)


def test_summarize_online_events_handles_tool_metrics():
    response_events = [
        {
            "response_payload": {
                "latency_ms": 200,
                "retrieval_used": True,
                "tool_calls": [{"name": "register_for_event_tool", "status": "success"}],
            }
        }
    ]
    feedback_events = [{"liked": "Dislike"}]

    summary = summarize_online_events(response_events, feedback_events)

    assert summary["tool_call_rate"] == 1.0
    assert summary["tool_success_rate"] == 1.0
    assert summary["disapproval_rate"] == 1.0
