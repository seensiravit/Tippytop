"""Typed results, failures, and strict parsing for the LLM protocol."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LLMResult:
    content: str
    usage: dict[str, int]
    requested_model: str = ""
    returned_model: str = ""
    response_id: str = ""
    finish_reason: str = ""


class GenerationFailure(ValueError):
    """A rejected generation together with every raw response received."""

    def __init__(self, message: str, responses: list[LLMResult]):
        super().__init__(message)
        self.responses = responses


class LLMTransportFailure(ConnectionError):
    """A retryable endpoint failure that must not consume an experiment iteration."""

    def __init__(self, message: str, responses: list[LLMResult]):
        super().__init__(message)
        self.responses = responses


class LLMDeadlineExceeded(TimeoutError):
    pass
