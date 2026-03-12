from __future__ import annotations

from typing import Any, Iterable, Mapping, Protocol, Sequence

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from .contracts import ChatTurn, ToolExecutionRecord


class ToolLike(Protocol):
    name: str

    def invoke(self, input: Any, config: Any | None = None, **kwargs: Any) -> Any: ...


def history_to_chat_turns(
    history: Sequence[ChatTurn | tuple[str, str]] | None,
) -> list[ChatTurn]:
    turns: list[ChatTurn] = []
    for item in history or ():
        if isinstance(item, ChatTurn):
            turns.append(item)
            continue
        if not isinstance(item, tuple) or len(item) != 2:
            raise TypeError("History items must be ChatTurn instances or (user, assistant) tuples.")
        turns.append(ChatTurn(user=item[0], assistant=item[1]))
    return turns


def build_messages(
    *,
    system_prompt: str,
    question: str,
    history: Sequence[ChatTurn | tuple[str, str]] | None = None,
) -> list[Any]:
    messages: list[Any] = [SystemMessage(content=system_prompt)]
    for turn in history_to_chat_turns(history):
        if turn.user:
            messages.append(HumanMessage(content=turn.user))
        if turn.assistant:
            messages.append(AIMessage(content=turn.assistant))
    messages.append(HumanMessage(content=question))
    return messages


def make_tool_registry(tools: Iterable[ToolLike]) -> dict[str, ToolLike]:
    return {tool.name: tool for tool in tools}


def execute_tool_calls(
    tool_calls: Sequence[Mapping[str, Any]],
    *,
    tool_registry: Mapping[str, ToolLike],
) -> tuple[list[ToolMessage], list[ToolExecutionRecord]]:
    tool_messages: list[ToolMessage] = []
    execution_records: list[ToolExecutionRecord] = []

    for tool_call in tool_calls:
        tool_name = str(tool_call.get("name") or "")
        tool_args = dict(tool_call.get("args") or {})
        tool_call_id = str(tool_call.get("id") or "")
        tool = tool_registry.get(tool_name)

        if tool is None:
            error_message = f"Tool error ({tool_name}): unknown tool."
            execution_records.append(
                ToolExecutionRecord(
                    name=tool_name,
                    args=tool_args,
                    status="error",
                    tool_call_id=tool_call_id,
                    error=error_message,
                )
            )
            tool_messages.append(
                ToolMessage(
                    content=error_message,
                    tool_call_id=tool_call_id,
                    name=tool_name or "unknown_tool",
                )
            )
            continue

        try:
            result = tool.invoke(tool_args)
            result_text = coerce_text(result)
            execution_records.append(
                ToolExecutionRecord(
                    name=tool_name,
                    args=tool_args,
                    status="success",
                    tool_call_id=tool_call_id,
                    result=result_text,
                )
            )
            tool_messages.append(
                ToolMessage(
                    content=result_text,
                    tool_call_id=tool_call_id,
                    name=tool_name,
                )
            )
        except Exception as exc:
            error_message = f"Tool error ({tool_name}): {exc}"
            execution_records.append(
                ToolExecutionRecord(
                    name=tool_name,
                    args=tool_args,
                    status="error",
                    tool_call_id=tool_call_id,
                    error=error_message,
                )
            )
            tool_messages.append(
                ToolMessage(
                    content=error_message,
                    tool_call_id=tool_call_id,
                    name=tool_name,
                )
            )

    return tool_messages, execution_records


def coerce_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if isinstance(item, dict):
                text = item.get("text")
                if text:
                    parts.append(str(text))
                    continue
                if item.get("type") == "text" and item.get("text"):
                    parts.append(str(item["text"]))
                    continue
            parts.append(str(item))
        return "\n".join(part for part in parts if part).strip()
    return str(content)
