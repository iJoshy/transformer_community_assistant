from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ChatTurn:
    user: str
    assistant: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ToolExecutionRecord:
    name: str
    args: dict[str, Any]
    status: str
    tool_call_id: str = ""
    result: str = ""
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class AssistantResponse:
    query: str
    answer: str
    retrieved_context: str
    retrieved_source_ids: tuple[str, ...]
    retrieved_chunks: tuple[dict[str, Any], ...]
    retrieval_used: bool
    tool_called: bool
    tool_calls: tuple[ToolExecutionRecord, ...]
    latency_ms: int
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["retrieved_source_ids"] = list(self.retrieved_source_ids)
        payload["retrieved_chunks"] = list(self.retrieved_chunks)
        payload["tool_calls"] = [item.to_dict() for item in self.tool_calls]
        return payload
