from __future__ import annotations

import gradio as gr

from app import ChatSessionController, build_demo
from app.controller import (
    chatbot_messages_to_turns,
    default_status_message,
    format_response_status,
)
from assistant import AssistantResponse, ToolExecutionRecord
from evals import OnlineFeedbackLogger


class StubAssistantService:
    def __init__(self, response: AssistantResponse):
        self.response = response
        self.calls: list[dict[str, object]] = []

    def answer(self, question: str, *, history, k: int, max_chars: int) -> AssistantResponse:
        self.calls.append(
            {
                "question": question,
                "history": history,
                "k": k,
                "max_chars": max_chars,
            }
        )
        return self.response


def make_response(*, answer: str = "Community Meetup is on April 2.") -> AssistantResponse:
    return AssistantResponse(
        query="What events are coming up?",
        answer=answer,
        retrieved_context="Community Meetup is on April 2.",
        retrieved_source_ids=("event-1",),
        retrieved_chunks=(
            {
                "content": "Community Meetup is on April 2.",
                "metadata": {"source_id": "event-1", "chunk_id": "event-1#chunk-0"},
                "source_id": "event-1",
            },
        ),
        retrieval_used=True,
        tool_called=True,
        tool_calls=(
            ToolExecutionRecord(
                name="register_for_event_tool",
                args={"email": "user@example.com", "event_id": "event-1"},
                status="success",
                result="Registration saved and confirmation email sent.",
            ),
        ),
        latency_ms=218,
    )


def test_chatbot_messages_to_turns_keeps_completed_pairs_only():
    history = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi"},
        {"role": "user", "content": "What events are coming up?"},
    ]

    turns = chatbot_messages_to_turns(history)

    assert len(turns) == 1
    assert turns[0].user == "Hello"
    assert turns[0].assistant == "Hi"


def test_controller_handles_message_and_formats_status():
    response = make_response()
    service = StubAssistantService(response)
    controller = ChatSessionController(assistant_service=service, k=6, max_chars=3200)

    history, status, details, session_state, feedback_status = controller.handle_message(
        "Register me for Community Meetup.",
        [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}],
    )

    assert history[-2]["content"] == "Register me for Community Meetup."
    assert history[-1]["content"] == "Community Meetup is on April 2."
    assert "Retrieval used: Yes" in status
    assert "register_for_event_tool" in status
    assert details["retrieved_source_ids"] == ["event-1"]
    assert isinstance(session_state, dict)
    assert "Feedback status" in feedback_status
    assert service.calls[0]["k"] == 6
    assert service.calls[0]["max_chars"] == 3200


def test_controller_rejects_empty_message():
    controller = ChatSessionController(assistant_service=StubAssistantService(make_response()))

    history, status, details, session_state, feedback_status = controller.handle_message("   ", [])

    assert history == []
    assert "Input needed" in status
    assert details["status"] == "No assistant response yet."
    assert isinstance(session_state, dict)
    assert "Feedback status" in feedback_status


def test_controller_logs_feedback_events(tmp_path):
    response = make_response()
    service = StubAssistantService(response)
    logger = OnlineFeedbackLogger(
        response_log_path=tmp_path / "responses.jsonl",
        feedback_log_path=tmp_path / "feedback.jsonl",
    )
    controller = ChatSessionController(
        assistant_service=service,
        online_feedback_logger=logger,
    )
    history, _, _, session_state, _ = controller.handle_message(
        "What events are coming up?",
        [],
    )

    like_data = gr.LikeData(None, {"index": len(history) - 1, "value": history[-1], "liked": True})
    updated_session_state, feedback_status = controller.handle_feedback(history, session_state, like_data)

    assert updated_session_state["session_id"] == session_state["session_id"]
    assert "Thumbs up recorded." in feedback_status


def test_build_demo_returns_gradio_blocks():
    controller = ChatSessionController(assistant_service=StubAssistantService(make_response()))

    demo = build_demo(controller)

    assert isinstance(demo, gr.Blocks)
    assert demo.title == "Transformer Community Assistant"
    assert default_status_message().startswith("### Session status")


def test_format_response_status_lists_sources_and_tool_calls():
    response = make_response()

    status = format_response_status(response)

    assert "Source IDs: `event-1`" in status
    assert "Tool calls: `register_for_event_tool` (success)" in status
