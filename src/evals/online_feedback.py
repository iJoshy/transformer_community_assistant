from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

from gradio import LikeData

from assistant import AssistantResponse

from .contracts import FeedbackLogEvent, ResponseLogEvent
from .storage import append_jsonl


DEFAULT_RESPONSE_LOG_PATH = Path("data/evals/response_events.jsonl")
DEFAULT_FEEDBACK_LOG_PATH = Path("data/evals/feedback_events.jsonl")


class OnlineFeedbackLogger:
    def __init__(
        self,
        *,
        response_log_path: str | Path = DEFAULT_RESPONSE_LOG_PATH,
        feedback_log_path: str | Path = DEFAULT_FEEDBACK_LOG_PATH,
    ):
        self.response_log_path = Path(response_log_path)
        self.feedback_log_path = Path(feedback_log_path)

    def new_session_state(self) -> dict[str, Any]:
        return {
            "session_id": uuid4().hex,
            "response_ids_by_index": {},
        }

    def ensure_session_state(self, session_state: Mapping[str, Any] | None) -> dict[str, Any]:
        return self._copy_state(session_state)

    def record_response(
        self,
        *,
        session_state: Mapping[str, Any] | None,
        user_message: str,
        assistant_message: str,
        assistant_message_index: int,
        response: AssistantResponse,
    ) -> dict[str, Any]:
        state = self._copy_state(session_state)
        response_id = uuid4().hex
        event = ResponseLogEvent.create(
            response_id=response_id,
            session_id=state["session_id"],
            assistant_message_index=assistant_message_index,
            user_message=user_message,
            assistant_message=assistant_message,
            response_payload=response.to_dict(),
        )
        append_jsonl(self.response_log_path, event.to_dict())
        response_ids_by_index = dict(state.get("response_ids_by_index") or {})
        response_ids_by_index[str(assistant_message_index)] = response_id
        state["response_ids_by_index"] = response_ids_by_index
        state["last_response_id"] = response_id
        state["last_assistant_message_index"] = assistant_message_index
        return state

    def record_feedback(
        self,
        *,
        session_state: Mapping[str, Any] | None,
        history: list[dict[str, Any]] | None,
        like_data: LikeData,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        state = self._copy_state(session_state)
        assistant_index = _coerce_feedback_index(like_data.index)
        response_id = str(
            (state.get("response_ids_by_index") or {}).get(str(assistant_index))
            or state.get("last_response_id")
            or ""
        )
        assistant_message = _resolve_feedback_message(history, like_data.value, assistant_index)
        event = FeedbackLogEvent.create(
            session_id=state["session_id"],
            response_id=response_id,
            assistant_message_index=assistant_index,
            liked=like_data.liked,
            assistant_message=assistant_message,
        )
        payload = event.to_dict()
        append_jsonl(self.feedback_log_path, payload)
        return state, payload

    def _copy_state(self, session_state: Mapping[str, Any] | None) -> dict[str, Any]:
        if not session_state:
            return self.new_session_state()
        state = dict(session_state)
        state["response_ids_by_index"] = dict(state.get("response_ids_by_index") or {})
        state.setdefault("session_id", uuid4().hex)
        return state


def format_feedback_status(event: Mapping[str, Any]) -> str:
    liked = event.get("liked")
    if liked is True or liked == "Like":
        verdict = "Thumbs up recorded."
    elif liked is False or liked == "Dislike":
        verdict = "Thumbs down recorded."
    else:
        verdict = f"Feedback recorded: {liked}"

    response_id = str(event.get("response_id") or "unmatched")
    assistant_index = event.get("assistant_message_index")
    return (
        "### Feedback status\n"
        f"- {verdict}\n"
        f"- Response ID: `{response_id}`\n"
        f"- Assistant message index: `{assistant_index}`"
    )


def default_feedback_status() -> str:
    return (
        "### Feedback status\n"
        "- Use the thumbs controls on assistant responses to create online eval signals."
    )


def _coerce_feedback_index(index: Any) -> int:
    if isinstance(index, tuple):
        for item in reversed(index):
            if isinstance(item, int):
                return item
        return -1
    if isinstance(index, int):
        return index
    return -1


def _resolve_feedback_message(
    history: list[dict[str, Any]] | None,
    fallback_value: Any,
    assistant_index: int,
) -> str:
    if history and 0 <= assistant_index < len(history):
        content = history[assistant_index].get("content")
        if content:
            return str(content)
    if isinstance(fallback_value, dict):
        content = fallback_value.get("content")
        if content:
            return str(content)
    return str(fallback_value or "")
