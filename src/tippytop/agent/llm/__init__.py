"""LLM client factory."""
from __future__ import annotations

from .base import BaseLLMClient, LLMResponse, LLMError
from .mock import MockLLMClient
from .gemini import GeminiClient

# gemini-2.5-flash-lite is retired for new API keys; the API recommends this one.
DEFAULT_GEMINI_MODEL = "gemini-3.5-flash-lite"


def build_llm_client(spec: str = "mock", **kwargs) -> BaseLLMClient:
    """spec: 'mock' | 'gemini'. Extra kwargs pass through to the client."""
    if spec == "mock":
        return MockLLMClient(kwargs.pop("scripts", []), **kwargs)
    if spec == "gemini":
        kwargs.setdefault("model", DEFAULT_GEMINI_MODEL)
        return GeminiClient(**kwargs)
    raise ValueError(f"unknown llm spec {spec!r} (expected 'mock' or 'gemini')")


__all__ = ["BaseLLMClient", "LLMResponse", "LLMError", "MockLLMClient",
           "GeminiClient", "build_llm_client", "DEFAULT_GEMINI_MODEL"]
