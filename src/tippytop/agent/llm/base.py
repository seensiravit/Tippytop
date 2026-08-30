"""LLM client abstraction. The loop only needs one call: generate(system, user)."""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


class LLMError(RuntimeError):
    """Raised when a provider call fails or returns no usable content.

    The orchestrator treats this as a failed iteration (recovery), never a crash.
    """


@dataclass
class LLMResponse:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    raw: dict = field(default_factory=dict)


class BaseLLMClient(ABC):
    model: str = "base"

    @abstractmethod
    def generate(self, system: str, user: str, *, temperature: float = 0.4,
                 max_output_tokens: int = 8192) -> LLMResponse:
        raise NotImplementedError
