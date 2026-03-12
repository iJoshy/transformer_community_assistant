from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any

from langchain_openai import ChatOpenAI, OpenAIEmbeddings


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENAI_PROVIDER = "openai"
OPENROUTER_PROVIDER = "openrouter"

DEFAULT_OPENAI_CHAT_MODEL = "gpt-4.1-nano"
DEFAULT_OPENROUTER_CHAT_MODEL = "openai/gpt-4.1-nano"
DEFAULT_OPENAI_EMBEDDING_MODEL = "text-embedding-3-large"
DEFAULT_OPENROUTER_EMBEDDING_MODEL = "openai/text-embedding-3-large"
DEFAULT_CHAT_MODEL = DEFAULT_OPENAI_CHAT_MODEL
DEFAULT_EMBEDDING_MODEL = DEFAULT_OPENAI_EMBEDDING_MODEL


@dataclass(frozen=True)
class OpenAICompatibleProvider:
    name: str
    api_key: str
    chat_model: str
    embedding_model: str
    base_url: str | None = None
    default_headers: dict[str, str] | None = None

    def chat_kwargs(
        self,
        *,
        model: str | None = None,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": model or self.chat_model,
            "temperature": temperature,
            "api_key": self.api_key,
        }
        if self.base_url:
            kwargs["base_url"] = self.base_url
        if self.default_headers:
            kwargs["default_headers"] = dict(self.default_headers)
        return kwargs

    def embedding_kwargs(self, *, model: str | None = None) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": model or self.embedding_model,
            "api_key": self.api_key,
        }
        if self.base_url:
            kwargs["base_url"] = self.base_url
        if self.default_headers:
            kwargs["default_headers"] = dict(self.default_headers)
        return kwargs


def get_default_chat_model(*, provider: str | None = None) -> str:
    selected_provider = provider or resolve_provider_name()
    if selected_provider == OPENROUTER_PROVIDER:
        return os.getenv("OPENROUTER_CHAT_MODEL", DEFAULT_OPENROUTER_CHAT_MODEL)
    return os.getenv("OPENAI_CHAT_MODEL", DEFAULT_OPENAI_CHAT_MODEL)


def get_default_embedding_model(*, provider: str | None = None) -> str:
    selected_provider = provider or resolve_provider_name()
    if selected_provider == OPENROUTER_PROVIDER:
        return os.getenv(
            "OPENROUTER_EMBEDDING_MODEL",
            DEFAULT_OPENROUTER_EMBEDDING_MODEL,
        )
    return os.getenv("OPENAI_EMBEDDING_MODEL", DEFAULT_OPENAI_EMBEDDING_MODEL)


def resolve_provider_name() -> str:
    if os.getenv("OPENROUTER_API_KEY"):
        return OPENROUTER_PROVIDER
    if os.getenv("OPEN_API_KEY") or os.getenv("OPENAI_API_KEY"):
        return OPENAI_PROVIDER
    return OPENAI_PROVIDER


def resolve_provider(
    *,
    chat_model: str | None = None,
    embedding_model: str | None = None,
) -> OpenAICompatibleProvider:
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    if openrouter_api_key:
        return OpenAICompatibleProvider(
            name=OPENROUTER_PROVIDER,
            api_key=openrouter_api_key,
            base_url=os.getenv("OPENROUTER_BASE_URL", OPENROUTER_BASE_URL),
            chat_model=chat_model or get_default_chat_model(provider=OPENROUTER_PROVIDER),
            embedding_model=embedding_model
            or get_default_embedding_model(provider=OPENROUTER_PROVIDER),
            default_headers=_resolve_openrouter_headers(),
        )

    openai_api_key = os.getenv("OPEN_API_KEY") or os.getenv("OPENAI_API_KEY")
    if openai_api_key:
        return OpenAICompatibleProvider(
            name=OPENAI_PROVIDER,
            api_key=openai_api_key,
            base_url=os.getenv("OPENAI_BASE_URL") or None,
            chat_model=chat_model or get_default_chat_model(provider=OPENAI_PROVIDER),
            embedding_model=embedding_model
            or get_default_embedding_model(provider=OPENAI_PROVIDER),
        )

    raise RuntimeError(
        "No model provider is configured. Set OPENROUTER_API_KEY or OPEN_API_KEY/OPENAI_API_KEY."
    )


def build_chat_model(
    *,
    model: str | None = None,
    temperature: float = 0.0,
) -> ChatOpenAI:
    provider = resolve_provider(chat_model=model)
    return ChatOpenAI(**provider.chat_kwargs(model=model, temperature=temperature))


def build_embeddings(*, model: str | None = None) -> OpenAIEmbeddings:
    provider = resolve_provider(embedding_model=model)
    return OpenAIEmbeddings(**provider.embedding_kwargs(model=model))


def _resolve_openrouter_headers() -> dict[str, str]:
    headers: dict[str, str] = {}
    referer = os.getenv("OPENROUTER_HTTP_REFERER") or os.getenv("OPENROUTER_REFERER")
    app_title = os.getenv("OPENROUTER_APP_TITLE") or os.getenv("OPENROUTER_TITLE")

    if referer:
        headers["HTTP-Referer"] = referer
    if app_title:
        headers["X-Title"] = app_title
    return headers
