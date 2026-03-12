from __future__ import annotations

import pytest

from model_provider import (
    OPENAI_PROVIDER,
    OPENROUTER_BASE_URL,
    OPENROUTER_PROVIDER,
    build_chat_model,
    build_embeddings,
    resolve_provider,
)


def clear_provider_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in (
        "OPENROUTER_API_KEY",
        "OPENROUTER_BASE_URL",
        "OPENROUTER_CHAT_MODEL",
        "OPENROUTER_EMBEDDING_MODEL",
        "OPENROUTER_HTTP_REFERER",
        "OPENROUTER_REFERER",
        "OPENROUTER_APP_TITLE",
        "OPENROUTER_TITLE",
        "OPEN_API_KEY",
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "OPENAI_CHAT_MODEL",
        "OPENAI_EMBEDDING_MODEL",
    ):
        monkeypatch.delenv(key, raising=False)


def test_resolve_provider_prefers_openrouter_when_present(monkeypatch: pytest.MonkeyPatch):
    clear_provider_env(monkeypatch)
    monkeypatch.setenv("OPENROUTER_API_KEY", "router-key")
    monkeypatch.setenv("OPEN_API_KEY", "openai-key")
    monkeypatch.setenv("OPENROUTER_HTTP_REFERER", "https://example.com")
    monkeypatch.setenv("OPENROUTER_APP_TITLE", "Transformer Assistant")

    provider = resolve_provider()

    assert provider.name == OPENROUTER_PROVIDER
    assert provider.api_key == "router-key"
    assert provider.base_url == OPENROUTER_BASE_URL
    assert provider.chat_model == "openai/gpt-4.1-nano"
    assert provider.embedding_model == "openai/text-embedding-3-large"
    assert provider.default_headers == {
        "HTTP-Referer": "https://example.com",
        "X-Title": "Transformer Assistant",
    }


def test_resolve_provider_accepts_open_api_key_alias(monkeypatch: pytest.MonkeyPatch):
    clear_provider_env(monkeypatch)
    monkeypatch.setenv("OPEN_API_KEY", "openai-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.example/v1")

    provider = resolve_provider()

    assert provider.name == OPENAI_PROVIDER
    assert provider.api_key == "openai-key"
    assert provider.base_url == "https://api.openai.example/v1"
    assert provider.chat_model == "gpt-4.1-nano"
    assert provider.embedding_model == "text-embedding-3-large"


def test_build_chat_model_uses_provider_configuration(monkeypatch: pytest.MonkeyPatch):
    clear_provider_env(monkeypatch)
    monkeypatch.setenv("OPENROUTER_API_KEY", "router-key")

    captured: dict[str, object] = {}

    class StubChatOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr("model_provider.ChatOpenAI", StubChatOpenAI)

    build_chat_model(model=None, temperature=0.3)

    assert captured["api_key"] == "router-key"
    assert captured["base_url"] == OPENROUTER_BASE_URL
    assert captured["model"] == "openai/gpt-4.1-nano"
    assert captured["temperature"] == 0.3


def test_build_embeddings_uses_provider_configuration(monkeypatch: pytest.MonkeyPatch):
    clear_provider_env(monkeypatch)
    monkeypatch.setenv("OPENROUTER_API_KEY", "router-key")

    captured: dict[str, object] = {}

    class StubEmbeddings:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr("model_provider.OpenAIEmbeddings", StubEmbeddings)

    build_embeddings()

    assert captured["api_key"] == "router-key"
    assert captured["base_url"] == OPENROUTER_BASE_URL
    assert captured["model"] == "openai/text-embedding-3-large"
