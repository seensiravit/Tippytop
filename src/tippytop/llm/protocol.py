"""Typed results, failures, and strict parsing for the LLM protocol."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..generated import GeneratedExperiment, parse_json_object


REVIEW_CHECKS = (
    "plan_fidelity",
    "substantive_novelty",
    "feature_path_complete",
    "continuous_ranking_scores",
    "leakage_safe",
    "resource_feasible",
)


@dataclass(frozen=True)
class LLMResult:
    content: str
    usage: dict[str, int]
    requested_model: str = ""
    returned_model: str = ""
    response_id: str = ""
    finish_reason: str = ""


@dataclass(frozen=True)
class ExperimentReview:
    verdict: str
    critique: str
    checks: dict[str, bool]
    experiment: GeneratedExperiment

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "critique": self.critique,
            "checks": self.checks,
            "experiment": self.experiment.to_dict(),
            "source_hash": self.experiment.source_hash,
        }


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


def parse_review(content: str) -> ExperimentReview:
    payload = parse_json_object(content)
    expected = {"verdict", "critique", "checks", "hypothesis", "expected_effect", "source"}
    missing = expected - set(payload)
    unknown = set(payload) - expected
    if missing:
        raise ValueError(f"experiment review is missing fields: {sorted(missing)}")
    if unknown:
        raise ValueError(f"experiment review has unknown fields: {sorted(unknown)}")

    verdict = payload["verdict"]
    critique = payload["critique"]
    if verdict not in {"pass", "revise"}:
        raise ValueError("review verdict must be 'pass' or 'revise'")
    if not isinstance(critique, str) or not critique.strip():
        raise ValueError("review critique must be a non-empty string")

    raw_checks = payload["checks"]
    if not isinstance(raw_checks, dict) or set(raw_checks) != set(REVIEW_CHECKS):
        raise ValueError(f"review checks must contain exactly: {list(REVIEW_CHECKS)}")
    if any(value is not True for value in raw_checks.values()):
        raise ValueError("every review check must pass for the final returned module")

    experiment = GeneratedExperiment.from_dict(
        {key: payload[key] for key in ("hypothesis", "expected_effect", "source")}
    )
    return ExperimentReview(
        verdict=verdict,
        critique=critique.strip(),
        checks={name: True for name in REVIEW_CHECKS},
        experiment=experiment,
    )
