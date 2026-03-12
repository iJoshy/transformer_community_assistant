from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True, slots=True)
class OfflineEvalCase:
    id: str
    category: str
    user_query: str
    expected_source_ids: tuple[str, ...] = ()
    expected_answer_points: tuple[str, ...] = ()
    forbidden_answer_points: tuple[str, ...] = ()
    expected_tool: str = ""
    expected_tool_args: dict[str, Any] = field(default_factory=dict)
    notes: str = ""

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "OfflineEvalCase":
        return cls(
            id=str(payload["id"]),
            category=str(payload["category"]),
            user_query=str(payload["user_query"]),
            expected_source_ids=tuple(payload.get("expected_source_ids") or ()),
            expected_answer_points=tuple(payload.get("expected_answer_points") or ()),
            forbidden_answer_points=tuple(payload.get("forbidden_answer_points") or ()),
            expected_tool=str(payload.get("expected_tool") or ""),
            expected_tool_args=dict(payload.get("expected_tool_args") or {}),
            notes=str(payload.get("notes") or ""),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["expected_source_ids"] = list(self.expected_source_ids)
        payload["expected_answer_points"] = list(self.expected_answer_points)
        payload["forbidden_answer_points"] = list(self.forbidden_answer_points)
        return payload


@dataclass(frozen=True, slots=True)
class OfflineEvalResult:
    case_id: str
    category: str
    query: str
    retrieval_relevant: bool
    answer_relevant: bool
    faithful: bool
    tool_correct: bool
    end_to_end_success: bool
    actual_source_ids: tuple[str, ...]
    actual_tool_names: tuple[str, ...]
    latency_ms: int
    answer: str

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["actual_source_ids"] = list(self.actual_source_ids)
        payload["actual_tool_names"] = list(self.actual_tool_names)
        return payload


@dataclass(frozen=True, slots=True)
class ResponseLogEvent:
    response_id: str
    session_id: str
    created_at: str
    assistant_message_index: int
    user_message: str
    assistant_message: str
    response_payload: dict[str, Any]

    @classmethod
    def create(
        cls,
        *,
        response_id: str,
        session_id: str,
        assistant_message_index: int,
        user_message: str,
        assistant_message: str,
        response_payload: dict[str, Any],
    ) -> "ResponseLogEvent":
        return cls(
            response_id=response_id,
            session_id=session_id,
            created_at=utc_now_iso(),
            assistant_message_index=assistant_message_index,
            user_message=user_message,
            assistant_message=assistant_message,
            response_payload=response_payload,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class FeedbackLogEvent:
    session_id: str
    response_id: str
    created_at: str
    assistant_message_index: int
    liked: bool | str
    assistant_message: str

    @classmethod
    def create(
        cls,
        *,
        session_id: str,
        response_id: str,
        assistant_message_index: int,
        liked: bool | str,
        assistant_message: str,
    ) -> "FeedbackLogEvent":
        return cls(
            session_id=session_id,
            response_id=response_id,
            created_at=utc_now_iso(),
            assistant_message_index=assistant_message_index,
            liked=liked,
            assistant_message=assistant_message,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
