from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from assistant import AssistantResponse, ChatTurn
from evals import (
    OnlineFeedbackLogger,
    default_feedback_status,
    format_feedback_status,
)


MessagePayload = dict[str, Any]


@dataclass(slots=True)
class ChatSessionController:
    assistant_service: Any
    k: int = 4
    max_chars: int = 4000
    online_feedback_logger: OnlineFeedbackLogger | None = None

    def handle_message(
        self,
        message: str,
        history: Sequence[MessagePayload] | None,
        session_state: dict[str, Any] | None = None,
    ) -> tuple[list[MessagePayload], str, dict[str, Any], dict[str, Any], str]:
        normalized_history = normalize_chat_messages(history)
        cleaned_message = (message or "").strip()
        if not cleaned_message:
            return (
                normalized_history,
                validation_status_message(),
                default_response_details(),
                self._ensure_session_state(session_state),
                default_feedback_status(),
            )

        assistant_history = chatbot_messages_to_turns(normalized_history)
        response = self.assistant_service.answer(
            cleaned_message,
            history=assistant_history,
            k=self.k,
            max_chars=self.max_chars,
        )
        assistant_text = (response.answer or "").strip() or response.error or fallback_error_text()
        updated_history = [
            *normalized_history,
            {"role": "user", "content": cleaned_message},
            {"role": "assistant", "content": assistant_text},
        ]
        updated_session_state = self._ensure_session_state(session_state)
        if self.online_feedback_logger is not None:
            updated_session_state = self.online_feedback_logger.record_response(
                session_state=updated_session_state,
                user_message=cleaned_message,
                assistant_message=assistant_text,
                assistant_message_index=len(updated_history) - 1,
                response=response,
            )
        return (
            updated_history,
            format_response_status(response),
            response.to_dict(),
            updated_session_state,
            default_feedback_status(),
        )

    def handle_feedback(
        self,
        history: Sequence[MessagePayload] | None,
        session_state: dict[str, Any] | None,
        like_data: Any,
    ) -> tuple[dict[str, Any], str]:
        if self.online_feedback_logger is None:
            return self._ensure_session_state(session_state), default_feedback_status()
        updated_session_state, payload = self.online_feedback_logger.record_feedback(
            session_state=session_state,
            history=normalize_chat_messages(history),
            like_data=like_data,
        )
        return updated_session_state, format_feedback_status(payload)

    def reset(self) -> tuple[list[MessagePayload], str, dict[str, Any], dict[str, Any], str]:
        return (
            [],
            default_status_message(),
            default_response_details(),
            self._ensure_session_state(None),
            default_feedback_status(),
        )

    def _ensure_session_state(self, session_state: dict[str, Any] | None) -> dict[str, Any]:
        if self.online_feedback_logger is None:
            return dict(session_state or {})
        return self.online_feedback_logger.ensure_session_state(session_state)


def normalize_chat_messages(
    history: Sequence[MessagePayload] | None,
) -> list[MessagePayload]:
    normalized: list[MessagePayload] = []
    for item in history or ():
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "").strip()
        content = str(item.get("content") or "").strip()
        if role not in {"user", "assistant"} or not content:
            continue
        normalized.append({"role": role, "content": content})
    return normalized


def chatbot_messages_to_turns(
    history: Sequence[MessagePayload] | None,
) -> list[ChatTurn]:
    turns: list[ChatTurn] = []
    pending_user: str | None = None

    for item in normalize_chat_messages(history):
        role = item["role"]
        content = item["content"]
        if role == "user":
            pending_user = content
            continue
        if role == "assistant" and pending_user is not None:
            turns.append(ChatTurn(user=pending_user, assistant=content))
            pending_user = None

    return turns


def default_status_message() -> str:
    return (
        "### Session status\n"
        "- Ready for a community question, event registration request, or registration lookup.\n"
        "- Retrieval and tool details will appear here after each response."
    )


def validation_status_message() -> str:
    return (
        "### Input needed\n"
        "- Enter a message before sending.\n"
        "- You can ask about community events, register for an event, or check registrations by email."
    )


def default_response_details() -> dict[str, Any]:
    return {"status": "No assistant response yet."}


def fallback_error_text() -> str:
    return "I could not produce a response for that request."


def format_response_status(response: AssistantResponse) -> str:
    lines = ["### Latest response"]
    lines.append(f"- Retrieval used: {'Yes' if response.retrieval_used else 'No'}")
    if response.retrieved_source_ids:
        source_ids = ", ".join(response.retrieved_source_ids)
        lines.append(f"- Source IDs: `{source_ids}`")
    else:
        lines.append("- Source IDs: none")

    if response.tool_calls:
        tool_summaries = ", ".join(
            f"`{record.name}` ({record.status})" for record in response.tool_calls
        )
        lines.append(f"- Tool calls: {tool_summaries}")
    else:
        lines.append("- Tool calls: none")

    lines.append(f"- Latency: `{response.latency_ms} ms`")
    if response.error:
        lines.append(f"- Error: {response.error}")
    return "\n".join(lines)
