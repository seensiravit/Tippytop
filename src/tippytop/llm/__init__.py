"""OpenAI-compatible autonomous research client and protocol types."""

from .client import LLMClient
from .protocol import (
    REVIEW_CHECKS,
    ExperimentReview,
    GenerationFailure,
    LLMDeadlineExceeded,
    LLMResult,
    LLMTransportFailure,
)

__all__ = [
    "REVIEW_CHECKS",
    "ExperimentReview",
    "GenerationFailure",
    "LLMClient",
    "LLMDeadlineExceeded",
    "LLMResult",
    "LLMTransportFailure",
]
