"""OpenAI-compatible autonomous research client and protocol types."""

from .client import LLMClient
from .protocol import (
    GenerationFailure,
    LLMDeadlineExceeded,
    LLMResult,
    LLMTransportFailure,
)

__all__ = [
    "GenerationFailure",
    "LLMClient",
    "LLMDeadlineExceeded",
    "LLMResult",
    "LLMTransportFailure",
]
