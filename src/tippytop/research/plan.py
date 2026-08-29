"""Strict, model-agnostic research plans produced before experiment code."""

from __future__ import annotations

import json
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
        if not isinstance(payload, dict):
            raise ValueError("research plan must be an object")

        expected = {
            "hypothesis",
            "expected_effect",
            "rationale",
            "departure_from_prior_work",
            "model_and_objective",
            "implementation_outline",
            "failure_modes",
        }
        payload = _unwrap_plan(payload, expected)
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
            if isinstance(value, str) and value.strip():
                scalars[name] = value.strip()
            elif name == "model_and_objective" and isinstance(value, (dict, list)) and value:
                scalars[name] = json.dumps(value, sort_keys=True)
            else:
                raise ValueError(f"research plan field {name!r} must be a non-empty string")

        lists = {
            name: _string_tuple(payload[name], name)
            for name in ("implementation_outline", "failure_modes")
        }
        data_and_features = payload.get("data_and_features")
        lists["data_and_features"] = (
            ()
            if data_and_features in (None, [])
            else _string_tuple(data_and_features, "data_and_features")
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


def _unwrap_plan(payload: dict[str, Any], expected: set[str]) -> dict[str, Any]:
    """Accept harmless response envelopes without weakening the plan itself."""

    if expected <= set(payload):
        return payload
    for key in ("research_plan", "experiment_plan", "plan"):
        nested = payload.get(key)
        if isinstance(nested, dict) and expected <= set(nested):
            return nested
    if len(payload) == 1:
        nested = next(iter(payload.values()))
        if isinstance(nested, dict) and expected <= set(nested):
            return nested
    return payload
