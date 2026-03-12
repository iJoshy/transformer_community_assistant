from __future__ import annotations

from langchain_core.messages import AIMessage

from assistant import CommunityAssistantService, build_system_prompt
from rag import RetrievalResult, RetrievedChunk


class StubRetriever:
    def __init__(self, result: RetrievalResult):
        self.result = result
        self.calls: list[tuple[str, int, int]] = []

    def retrieve(self, query: str, *, k: int = 4, max_chars: int = 4000) -> RetrievalResult:
        self.calls.append((query, k, max_chars))
        return self.result


class StubLLM:
    def __init__(self, responses: list[AIMessage]):
        self.responses = responses
        self.messages: list[list[object]] = []
        self.bound_tools = None

    def bind_tools(self, tools):
        self.bound_tools = list(tools)
        return self

    def invoke(self, messages):
        self.messages.append(list(messages))
        if not self.responses:
            raise AssertionError("StubLLM received more invocations than expected.")
        return self.responses.pop(0)


class StubTool:
    def __init__(self, name: str, output: str):
        self.name = name
        self.output = output
        self.calls: list[dict[str, object]] = []

    def invoke(self, input, config=None, **kwargs):
        self.calls.append(dict(input))
        return self.output


def make_retrieval_result() -> RetrievalResult:
    return RetrievalResult(
        query="What events are coming up?",
        k=2,
        context="Community Meetup is on April 2.\n\nHack Night is on April 30.",
        chunks=(
            RetrievedChunk(
                content="Community Meetup is on April 2.",
                metadata={"source_id": "event-1", "chunk_id": "event-1#chunk-0"},
            ),
            RetrievedChunk(
                content="Hack Night is on April 30.",
                metadata={"source_id": "event-2", "chunk_id": "event-2#chunk-0"},
            ),
        ),
    )


def test_build_system_prompt_includes_context_and_tool_rules():
    prompt = build_system_prompt(context="Community Meetup is on April 2.")

    assert "Today's date is:" in prompt
    assert "Never call a tool if required parameters are missing." in prompt
    assert "Community Meetup is on April 2." in prompt


def test_service_answers_question_using_retrieval_context():
    retriever = StubRetriever(make_retrieval_result())
    llm = StubLLM([AIMessage(content="Community Meetup is on April 2.")])
    service = CommunityAssistantService(
        retriever=retriever,
        llm=llm,
        tools=[],
        bind_tools=False,
    )

    response = service.answer(
        "What events are coming up?",
        history=[("Hello", "Hi there.")],
    )

    assert response.answer == "Community Meetup is on April 2."
    assert response.retrieval_used is True
    assert response.retrieved_source_ids == ("event-1", "event-2")
    assert response.tool_called is False
    assert response.error == ""
    assert retriever.calls == [("What events are coming up?", 4, 4000)]
    assert llm.messages[0][0].content.startswith("You are a knowledgeable, friendly assistant")


def test_service_executes_tool_calls_and_returns_final_answer():
    retriever = StubRetriever(make_retrieval_result())
    llm = StubLLM(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "register_for_event_tool",
                        "args": {"email": "user@test.com", "event_id": "event-1"},
                        "id": "call_1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="You are registered for Community Meetup."),
        ]
    )
    register_tool = StubTool(
        name="register_for_event_tool",
        output="Registration saved and confirmation email sent.",
    )
    service = CommunityAssistantService(
        retriever=retriever,
        llm=llm,
        tools=[register_tool],
    )

    response = service.answer("Register me for Community Meetup with email user@test.com")

    assert register_tool.calls == [{"email": "user@test.com", "event_id": "event-1"}]
    assert response.answer == "You are registered for Community Meetup."
    assert response.tool_called is True
    assert response.tool_calls[0].name == "register_for_event_tool"
    assert response.tool_calls[0].status == "success"
    assert response.tool_calls[0].result == "Registration saved and confirmation email sent."
    assert llm.bound_tools is not None


def test_service_records_tool_errors_and_continues():
    class BrokenTool(StubTool):
        def invoke(self, input, config=None, **kwargs):
            raise RuntimeError("boom")

    retriever = StubRetriever(make_retrieval_result())
    llm = StubLLM(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "register_for_event_tool",
                        "args": {"email": "user@test.com", "event_id": "event-1"},
                        "id": "call_1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="I could not complete the registration."),
        ]
    )
    service = CommunityAssistantService(
        retriever=retriever,
        llm=llm,
        tools=[BrokenTool(name="register_for_event_tool", output="")],
    )

    response = service.answer("Register me for Community Meetup with email user@test.com")

    assert response.answer == "I could not complete the registration."
    assert response.tool_called is True
    assert response.tool_calls[0].status == "error"
    assert "Tool error (register_for_event_tool): boom" in response.tool_calls[0].error


def test_service_from_env_uses_dynamic_chat_builder(monkeypatch):
    retriever = StubRetriever(make_retrieval_result())
    captured: dict[str, object] = {}

    class StubBoundLLM(StubLLM):
        def __init__(self):
            super().__init__([AIMessage(content="Community Meetup is on April 2.")])

    def fake_from_config(*, config):
        return retriever

    def fake_build_chat_model(*, model=None, temperature=0.0):
        captured["model"] = model
        captured["temperature"] = temperature
        return StubBoundLLM()

    monkeypatch.setattr("assistant.service.RagRetriever.from_config", fake_from_config)
    monkeypatch.setattr("assistant.service.build_chat_model", fake_build_chat_model)

    service = CommunityAssistantService.from_env(model="openai/gpt-4.1-nano")

    assert isinstance(service.retriever, StubRetriever)
    assert captured == {"model": "openai/gpt-4.1-nano", "temperature": 0.0}
