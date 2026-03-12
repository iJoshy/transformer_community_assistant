from __future__ import annotations

from time import perf_counter
from typing import Any, Sequence

from event_registration import EVENT_REGISTRATION_TOOLS
from model_provider import DEFAULT_CHAT_MODEL, build_chat_model
from rag import RagConfig, RagRetriever

from .contracts import AssistantResponse, ChatTurn, ToolExecutionRecord
from .orchestrator import build_messages, coerce_text, execute_tool_calls, make_tool_registry
from .prompts import build_system_prompt


class CommunityAssistantService:
    def __init__(
        self,
        *,
        retriever: RagRetriever,
        llm: Any,
        tools: Sequence[Any] | None = None,
        max_tool_rounds: int = 3,
        bind_tools: bool = True,
    ):
        self.retriever = retriever
        self.tools = tuple(tools or ())
        self.tool_registry = make_tool_registry(self.tools)
        self.max_tool_rounds = max_tool_rounds
        self.llm = self._prepare_llm(llm, bind_tools=bind_tools)

    @classmethod
    def from_openai(
        cls,
        *,
        rag_config: RagConfig | None = None,
        model: str | None = None,
        temperature: float = 0.0,
        tools: Sequence[Any] | None = None,
        max_tool_rounds: int = 3,
    ) -> "CommunityAssistantService":
        return cls.from_env(
            rag_config=rag_config,
            model=model,
            temperature=temperature,
            tools=tools,
            max_tool_rounds=max_tool_rounds,
        )

    @classmethod
    def from_env(
        cls,
        *,
        rag_config: RagConfig | None = None,
        model: str | None = None,
        temperature: float = 0.0,
        tools: Sequence[Any] | None = None,
        max_tool_rounds: int = 3,
    ) -> "CommunityAssistantService":
        retriever = RagRetriever.from_config(config=rag_config or RagConfig())
        llm = build_chat_model(model=model, temperature=temperature)
        return cls(
            retriever=retriever,
            llm=llm,
            tools=tools or EVENT_REGISTRATION_TOOLS,
            max_tool_rounds=max_tool_rounds,
            bind_tools=True,
        )

    def answer(
        self,
        question: str,
        *,
        history: Sequence[ChatTurn | tuple[str, str]] | None = None,
        k: int = 4,
        max_chars: int = 4000,
    ) -> AssistantResponse:
        started_at = perf_counter()
        retrieval = self.retriever.retrieve(question, k=k, max_chars=max_chars)
        system_prompt = build_system_prompt(context=retrieval.context)
        messages = build_messages(
            system_prompt=system_prompt,
            question=question,
            history=history,
        )

        response = self.llm.invoke(messages)
        tool_records: list[ToolExecutionRecord] = []
        rounds = 0

        while getattr(response, "tool_calls", None):
            rounds += 1
            if rounds > self.max_tool_rounds:
                tool_records.append(
                    ToolExecutionRecord(
                        name="tool_loop_guard",
                        args={},
                        status="error",
                        error="Maximum tool-calling rounds exceeded.",
                    )
                )
                break

            messages.append(response)
            tool_messages, execution_records = execute_tool_calls(
                response.tool_calls,
                tool_registry=self.tool_registry,
            )
            tool_records.extend(execution_records)
            messages.extend(tool_messages)
            response = self.llm.invoke(messages)

        answer_text = coerce_text(getattr(response, "content", ""))
        if not answer_text and tool_records:
            last_success = next(
                (record for record in reversed(tool_records) if record.result),
                None,
            )
            if last_success is not None:
                answer_text = last_success.result

        error = ""
        if rounds > self.max_tool_rounds:
            error = "Maximum tool-calling rounds exceeded."

        latency_ms = int((perf_counter() - started_at) * 1000)
        return AssistantResponse(
            query=question,
            answer=answer_text,
            retrieved_context=retrieval.context,
            retrieved_source_ids=tuple(retrieval.source_ids),
            retrieved_chunks=tuple(chunk.to_dict() for chunk in retrieval.chunks),
            retrieval_used=bool(retrieval.context.strip()),
            tool_called=bool(tool_records),
            tool_calls=tuple(tool_records),
            latency_ms=latency_ms,
            error=error,
        )

    def _prepare_llm(self, llm: Any, *, bind_tools: bool) -> Any:
        if bind_tools and self.tools and hasattr(llm, "bind_tools"):
            return llm.bind_tools(list(self.tools))
        return llm
