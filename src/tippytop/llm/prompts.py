"""Message construction for the scientist, coder, executor, and repair stages."""

from __future__ import annotations

import ast
import json
import re
from typing import Any

from ..generated import GeneratedExperiment
from ..research import ResearchPlan
from ..research.contract import experiment_contract, research_environment


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
                "signal becomes valid prediction-time state. Treat structured recovery diagnostics from "
                "prior attempts as established environment facts and do not repeat those failures. "
                "Return one JSON object only."
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
                        "failure_modes": ["risk the coder and runtime repair loop must address"],
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
                "starter model or redesign the plan. Respect all prior recovery diagnostics in the research "
                "context. Host helpers are optional, not a prescribed architecture. "
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
    failure_history: list[dict[str, Any]] | None = None,
) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You are repairing an ML experiment that failed in an isolated runtime. "
                "Return one JSON object with exactly an edits array. Each edit must contain an exact, "
                "unique old substring copied from the failed source and its new replacement string. "
                "Make the smallest executable correction that fixes the exact traceback; do not "
                "redesign the model or introduce unrelated changes. Preserve the research intent, "
                "and do not regenerate the complete module. Comment-only changes are not a repair. "
                "Estimator hyperparameters such as objective, metric, "
                "n_estimators, and learning_rate belong in the estimator constructor or training "
                "parameter dictionary, not a scikit-learn estimator's fit call. Review the complete "
                "failure history and fix every recurring mistake in one pass; a repair that only removes "
                "the newest invalid keyword while retaining earlier API misuse is invalid. Treat supplied "
                "installed signatures as authoritative. Do not include Markdown."
            ),
        },
        {
            "role": "user",
            "content": _json(
                {
                    "response_schema": {
                        "edits": [
                            {
                                "old": "exact unique substring from failed source",
                                "new": "replacement substring",
                            }
                        ],
                    },
                    "repair_constraints": {
                        "required_functions": experiment_contract()["required_functions"],
                        "preserve_research_intent": True,
                        "source_must_remain_validated_and_pickleable": True,
                        "predict_must_return_continuous_finite_scores": True,
                    },
                    "research_plan": _repair_plan(plan),
                    "failed_experiment": {
                        "hypothesis": failed.hypothesis,
                        "expected_effect": failed.expected_effect,
                        "source_hash": failed.source_hash,
                    },
                    "failed_source_excerpts": _source_excerpts(failed.source, error),
                    "runtime_error": _concise_error(error),
                    "cumulative_runtime_failures": _compact_failures(failure_history or []),
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


def _repair_plan(plan: ResearchPlan | None) -> dict[str, Any] | None:
    if plan is None:
        return None
    return {
        "hypothesis": plan.hypothesis,
        "model_and_objective": plan.model_and_objective,
        "failure_modes": list(plan.failure_modes),
    }


def _source_excerpts(source: str, error: str) -> list[dict[str, Any]]:
    lines = source.splitlines()
    failing_lines = sorted(
        {
            int(match)
            for match in re.findall(r'experiment\.py", line (\d+)', error)
            if 1 <= int(match) <= len(lines)
        }
    )
    ranges = [(0, min(25, len(lines)))]
    ranges.extend((max(0, line - 21), min(len(lines), line + 20)) for line in failing_lines)
    ranges.extend(_stateful_function_ranges(source, {"__init__", "fit", "predict"}))
    merged: list[tuple[int, int]] = []
    for start, end in sorted(ranges):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    excerpts = []
    remaining_chars = 24_000
    for start, end in merged:
        rendered = "\n".join(lines[start:end])
        if not rendered or remaining_chars <= 0:
            continue
        rendered = rendered[:remaining_chars]
        excerpts.append({"start_line": start + 1, "source": rendered})
        remaining_chars -= len(rendered)
    if failing_lines:
        return excerpts
    if len(source) <= 12_000:
        return [{"start_line": 1, "source": source}]
    return [
        {"start_line": 1, "source": source[:4000]},
        {"start_line": None, "source": source[-8000:]},
    ]


def _stateful_function_ranges(source: str, names: set[str]) -> list[tuple[int, int]]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    line_count = len(source.splitlines())
    return [
        (max(0, node.lineno - 1), min(line_count, node.end_lineno or node.lineno))
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names
    ]


def _concise_error(error: str, limit: int = 3000) -> str:
    if len(error) <= limit:
        return error
    return f"{error[:500]}\n... traceback truncated ...\n{error[-2460:]}"


def _compact_failures(failures: list[dict[str, Any]]) -> list[dict[str, Any]]:
    signature_index = next(
        (
            index
            for index in range(len(failures) - 1, -1, -1)
            if isinstance(failures[index].get("diagnostics"), dict)
            and failures[index]["diagnostics"].get("installed_signatures")
        ),
        None,
    )
    compact: list[dict[str, Any]] = []
    for index, failure in enumerate(failures):
        diagnostics = dict(failure.get("diagnostics", {}))
        if index != signature_index:
            diagnostics.pop("installed_signatures", None)
        compact.append(
            {
                "experiment_id": failure.get("experiment_id"),
                "source_hash": failure.get("source_hash"),
                "error": _exception_summary(str(failure.get("error", ""))),
                "diagnostics": diagnostics,
            }
        )
    return compact


def _exception_summary(error: str) -> str:
    lines = [line.strip() for line in error.splitlines() if line.strip()]
    if not lines:
        return ""
    return lines[0] if len(lines) == 1 else f"{lines[0]}\n{lines[-1]}"
