"""Message construction for the scientist, coder, reviewer, and repair stages."""

from __future__ import annotations

import json
from typing import Any

from ..generated import GeneratedExperiment
from ..research import ResearchPlan
from ..research.contract import experiment_contract, research_environment
from .protocol import REVIEW_CHECKS


def research_messages(context: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You are the lead scientist in an autonomous recommender-system lab. Select one "
                "falsifiable, resource-feasible experiment, not a list of candidates and not code. "
                "Escape local helper-driven variations: use the raw schema and measured failures to "
                "make a substantive change in representation, supervision, objective, model, or a "
                "combination. Explain exactly how it departs from prior work and how every training-only "
                "signal becomes valid prediction-time state. Return one JSON object only."
            ),
        },
        {
            "role": "user",
            "content": _json(
                {
                    "research_plan_schema": {
                        "hypothesis": "specific falsifiable claim",
                        "expected_effect": "expected ranking-metric effect",
                        "rationale": "scientific mechanism",
                        "departure_from_prior_work": "why this is not another measured variation",
                        "data_and_features": ["precise signal and causal construction"],
                        "model_and_objective": "representation, learner, supervision, and objective",
                        "implementation_outline": ["ordered executable design step"],
                        "failure_modes": ["risk the coder and reviewer must address"],
                    },
                    "research_environment": research_environment(),
                    "research_context": context,
                }
            ),
        },
    ]


def generation_messages(
    context: dict[str, Any],
    plan: ResearchPlan,
) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You are the implementation engineer in an autonomous ML lab. Write the complete "
                "experiment module for the scientist's approved plan. Implement every planned feature, "
                "supervision signal, model component, and objective; do not simplify it into a familiar "
                "starter model or redesign the plan. Host helpers are optional, not a prescribed architecture. "
                "Return only one JSON object with no Markdown or commentary."
            ),
        },
        {
            "role": "user",
            "content": _json(
                {
                    "response_schema": {
                        "hypothesis": "implemented experiment hypothesis",
                        "expected_effect": "expected ranking effect",
                        "source": "complete Python module",
                    },
                    "experiment_contract": experiment_contract(),
                    "research_plan": plan.to_dict(),
                    "research_context": context,
                }
            ),
        },
    ]


def repair_messages(
    _context: dict[str, Any],
    failed: GeneratedExperiment,
    error: str,
    plan: ResearchPlan | None,
) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You are repairing an ML experiment that failed in an isolated runtime. "
                "Return one JSON object with exactly hypothesis, expected_effect, and source. "
                "Make the smallest executable correction that fixes the exact traceback; do not "
                "redesign the model or introduce unrelated changes. Preserve the research intent, "
                "and include the complete Python module as the source JSON string. Comment-only "
                "changes are not a repair. Estimator hyperparameters such as objective, metric, "
                "n_estimators, and learning_rate belong in the estimator constructor or training "
                "parameter dictionary, not a scikit-learn estimator's fit call. Do not include Markdown."
            ),
        },
        {
            "role": "user",
            "content": _json(
                {
                    "response_schema": {
                        "hypothesis": "repaired experiment hypothesis",
                        "expected_effect": "expected ranking effect",
                        "source": "complete repaired Python module",
                    },
                    "experiment_contract": experiment_contract(),
                    "research_plan": plan.to_dict() if plan is not None else None,
                    "failed_experiment": failed.to_dict(),
                    "runtime_error": error[-12000:],
                }
            ),
        },
    ]


def review_messages(
    context: dict[str, Any],
    plan: ResearchPlan,
    proposed: GeneratedExperiment,
) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You are the independent senior reviewer in an autonomous ML research loop. "
                "Inspect the proposed experiment against the scientist's plan, exact runtime contract, "
                "and measured history. Trace every planned feature into both final fit and predict matrices; "
                "reject ignored/no-op features, label leakage, class-label outputs instead of continuous "
                "ranking scores, undertrained one-step objectives, invalid helper use, non-vectorized "
                "full-data code, and repeated measured dead ends. If any blocking issue exists, rewrite "
                "the complete module yourself. Return one JSON object only with verdict ('pass' or "
                "'revise'), critique, all six boolean checks, hypothesis, expected_effect, and source. "
                "Every check must describe the final returned module."
            ),
        },
        {
            "role": "user",
            "content": _json(
                {
                    "response_schema": {
                        "verdict": "pass or revise",
                        "critique": "non-empty review of the final returned module",
                        "checks": {name: True for name in REVIEW_CHECKS},
                        "hypothesis": "final experiment hypothesis",
                        "expected_effect": "expected ranking effect",
                        "source": "complete final Python module",
                    },
                    "experiment_contract": experiment_contract(),
                    "research_plan": plan.to_dict(),
                    "research_context": context,
                    "proposed_experiment": proposed.to_dict(),
                }
            ),
        },
    ]


def reflection_messages(context: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You are an ML research agent reflecting on a measured experiment. "
                "State what the result supports, what failed, and the next direction in at most 180 words."
            ),
        },
        {"role": "user", "content": _json(context)},
    ]


def _json(value: dict[str, Any]) -> str:
    return json.dumps(value, sort_keys=True)
