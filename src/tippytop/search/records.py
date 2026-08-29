"""Audit-record formatting and measured-result diagnostics for search."""

from __future__ import annotations

import difflib
from typing import Any, Sequence

from ..artifacts import RunStore, source_revision
from ..llm import LLMClient, LLMDeadlineExceeded, LLMResult
from ..research import ResearchPlan


REFLECTION_CODE_CHARS = 6_000


def best_source(store: RunStore, state: dict[str, Any]) -> str:
    checkpoint = store.resolve_path(state["best"]["checkpoint"])
    source_path = checkpoint / "experiment.py" if checkpoint.is_dir() else None
    if source_path is None or not source_path.is_file():
        return ""
    return source_path.read_text(encoding="utf-8")


def source_diff(before: str, after: str, before_id: str, after_id: str) -> str:
    lines = difflib.unified_diff(
        before.splitlines(),
        after.splitlines(),
        fromfile=f"{before_id}/experiment.py",
        tofile=f"{after_id}/experiment.py",
        lineterm="",
    )
    rendered = "\n".join(lines)
    return rendered + ("\n" if rendered else "")


def generation_failure_record(
    iteration: int,
    parent_id: str,
    stage: str,
    error: Exception,
    research_plan: ResearchPlan | None,
    research_responses: Sequence[LLMResult],
    responses: Sequence[LLMResult],
    state: dict[str, Any],
) -> dict[str, Any]:
    return {
        "iteration": iteration,
        "id": f"iteration-{iteration:03d}",
        "parent_id": parent_id,
        "status": "generation_failed",
        "generation_stage": stage,
        "error": str(error),
        "became_best": False,
        "source_revision": source_revision(),
        "research_plan": research_plan.to_dict() if research_plan is not None else None,
        "research_responses": [response_dict(response) for response in research_responses],
        "responses": [response_dict(response) for response in responses],
        "manual_interventions": state["manual_interventions"],
    }


def reflect(
    llm: LLMClient,
    state: dict[str, Any],
    record: dict[str, Any],
    store: RunStore,
    best_before_experiment: dict[str, Any],
    deadline: float,
) -> LLMResult | None:
    try:
        return llm.reflect(
            {
                "baseline_validation": state["baseline_valid"],
                "best_before_experiment": best_before_experiment,
                "experiment": {
                    key: record.get(key)
                    for key in (
                        "hypothesis",
                        "expected_effect",
                        "source_hash",
                        "status",
                        "metrics",
                        "diagnostics",
                        "error",
                        "recovery",
                    )
                }
                | {
                    "code_diff": bounded_text(
                        str(record.get("code_diff") or ""),
                        REFLECTION_CODE_CHARS,
                    )
                },
            },
            deadline=deadline,
        )
    except (ConnectionError, LLMDeadlineExceeded) as error:
        store.event("reflection_failure", iteration=record["iteration"], error=str(error))
        return None


def validation_diagnostics(
    metrics: dict[str, Any],
    baseline: dict[str, Any],
    best_before: dict[str, Any],
    history: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    metric_names = ("GAUC", "nDCG@5", "primary")
    matching_iteration = None
    for prior in reversed(history):
        prior_metrics = prior.get("metrics")
        if not isinstance(prior_metrics, dict):
            continue
        if all(
            abs(float(metrics[name]) - float(prior_metrics.get(name, float("inf")))) <= 1e-12
            for name in metric_names
        ):
            matching_iteration = prior.get("iteration")
            break
    return {
        "primary_delta_from_baseline": float(metrics["primary"] - baseline["primary"]),
        "primary_delta_from_previous_best": float(metrics["primary"] - best_before["primary"]),
        "matching_prior_iteration": matching_iteration,
    }


def bounded_text(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    marker = "\n... <truncated for reflection> ...\n"
    prefix = (limit - len(marker)) // 2
    suffix = limit - len(marker) - prefix
    return value[:prefix] + marker + value[-suffix:]


def response_dict(response: LLMResult) -> dict[str, Any]:
    return {
        "content": response.content,
        "usage": response.usage,
        "requested_model": response.requested_model,
        "returned_model": response.returned_model,
        "response_id": response.response_id,
        "finish_reason": response.finish_reason,
    }


def add_usage(state: dict[str, Any], responses: Sequence[LLMResult]) -> None:
    for response in responses:
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            state["llm_usage"][key] += int(response.usage.get(key, 0))
