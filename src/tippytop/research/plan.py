"""Strict, model-agnostic research plans produced before experiment code."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ResearchPlan:
    hypothesis: str
    expected_effect: str
    rationale: str
    departure_from_prior_work: str
    data_and_features: tuple[str, ...]
    model_and_objective: str
    implementation_outline: tuple[str, ...]
    failure_modes: tuple[str, ...]

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ResearchPlan:
        expected = {
            "hypothesis",
            "expected_effect",
            "rationale",
            "departure_from_prior_work",
            "model_and_objective",
            "implementation_outline",
            "failure_modes",
        }
        missing = expected - set(payload)
        if missing:
            raise ValueError(f"research plan is missing fields: {sorted(missing)}")

        scalar_names = (
            "hypothesis",
            "expected_effect",
            "rationale",
            "departure_from_prior_work",
            "model_and_objective",
        )
        scalars: dict[str, str] = {}
        for name in scalar_names:
            value = payload[name]
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"research plan field {name!r} must be a non-empty string")
            scalars[name] = value.strip()

        lists = {
            name: _string_tuple(payload[name], name)
            for name in ("implementation_outline", "failure_modes")
        }
        lists["data_and_features"] = (
            _string_tuple(payload["data_and_features"], "data_and_features")
            if "data_and_features" in payload
            else ()
        )
        return cls(**scalars, **lists)

    def to_dict(self) -> dict[str, Any]:
        return {
            "hypothesis": self.hypothesis,
            "expected_effect": self.expected_effect,
            "rationale": self.rationale,
            "departure_from_prior_work": self.departure_from_prior_work,
            "data_and_features": list(self.data_and_features),
            "model_and_objective": self.model_and_objective,
            "implementation_outline": list(self.implementation_outline),
            "failure_modes": list(self.failure_modes),
        }


def _string_tuple(value: Any, name: str) -> tuple[str, ...]:
    if isinstance(value, str) and value.strip():
        return (value.strip(),)
    if not isinstance(value, list) or not value:
        raise ValueError(f"research plan field {name!r} must be text or a non-empty list")
    if any(not isinstance(item, str) or not item.strip() for item in value):
        raise ValueError(f"research plan field {name!r} must contain non-empty strings")
    return tuple(item.strip() for item in value)
