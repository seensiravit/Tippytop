"""One auditable scientist-coder-reviewer experiment iteration."""

from __future__ import annotations

import difflib
import time
from pathlib import Path
from typing import Any, Sequence

from .artifacts import (
    HostRevisionChanged,
    RunStore,
    assert_source_revision,
    atomic_write_text,
    read_jsonl,
    source_revision,
    utc_now,
)
from .config import RunConfig
from .convergence import ConvergenceTracker
from .generated import GeneratedExperiment
from .llm import (
    ExperimentReview,
    GenerationFailure,
    LLMClient,
    LLMDeadlineExceeded,
    LLMResult,
    LLMTransportFailure,
)
from .research_context import build_research_context
from .research_plan import ResearchPlan
from .runner import ExperimentFailure, run_experiment
from .search_journal import (
    begin_attempt as _begin_attempt,
    commit_iteration as _commit_iteration,
    finish_attempt as _finish_attempt,
    persist_state as _persist_state,
    update_attempt as _update_attempt,
)


MAX_RUNTIME_REPAIRS = 2
REFLECTION_CODE_CHARS = 6_000


def run_iteration(
    config: RunConfig,
    store: RunStore,
    state: dict[str, Any],
    tracker: ConvergenceTracker,
    llm: LLMClient,
    used_hashes: set[str],
    research_data_path: Path,
    wall_started: float,
    deadline: float,
) -> None:
    iteration = int(state["iteration"]) + 1
    history = read_jsonl(store.path / "iterations.jsonl")
    parent_id = str(state["best"]["id"])
    expected_revision = str(state["source_revision"])
    assert_source_revision(expected_revision)
    parent_source = _best_source(store, state)
    context = build_research_context(store, state, history, parent_source)
    best_before_experiment = dict(state["best"]["metrics"])
    responses: list[LLMResult] = []
    research_responses: list[LLMResult] = []
    review_responses: list[LLMResult] = []
    research_plan: ResearchPlan | None = None
    pre_execution_review: ExperimentReview | None = None
    generation_stage = "research"

    store.event("research_started", iteration=iteration, parent_id=parent_id)
    try:
        research_plan, research_responses = llm.research(context, deadline=deadline)
        _add_usage(state, research_responses)
        generation_stage = "coding"
        store.event(
            "coding_started",
            iteration=iteration,
            hypothesis=research_plan.hypothesis,
        )
        experiment, responses = llm.generate(context, research_plan, deadline=deadline)
        _add_usage(state, responses)
        if experiment.source_hash in used_hashes:
            raise ValueError(f"source hash {experiment.source_hash} was already evaluated")
        proposed_experiment = experiment
        generation_stage = "review"
        store.event("review_started", iteration=iteration, source_hash=experiment.source_hash)
        pre_execution_review, review_responses = llm.review(
            context,
            research_plan,
            proposed_experiment,
            deadline=deadline,
        )
        _add_usage(state, review_responses)
        experiment = pre_execution_review.experiment
        if experiment.source_hash in used_hashes:
            raise ValueError(f"reviewed source hash {experiment.source_hash} was already evaluated")
    except LLMDeadlineExceeded:
        state["stopping_reason"] = "wall_clock_limit"
        _persist_state(store, state, wall_started)
        return
    except LLMTransportFailure as error:
        _add_usage(state, error.responses)
        store.event("generation_transport_failure", iteration=iteration, error=str(error))
        _persist_state(store, state, wall_started)
        time.sleep(2)
        return
    except (ConnectionError, ValueError) as error:
        if isinstance(error, GenerationFailure):
            failed_responses = error.responses
            if generation_stage == "research":
                research_responses.extend(failed_responses)
            elif generation_stage == "review":
                review_responses.extend(failed_responses)
            else:
                responses.extend(failed_responses)
            _add_usage(state, failed_responses)
        store.event(
            "generation_failed",
            iteration=iteration,
            stage=generation_stage,
            error=str(error),
        )
        record = _generation_failure_record(
            iteration,
            parent_id,
            generation_stage,
            error,
            research_plan,
            research_responses,
            responses,
            review_responses,
            state,
        )
        state["iteration"] = iteration
        state["stagnant"] = tracker.stagnant
        _commit_iteration(store, state, record, wall_started)
        return

    used_hashes.add(proposed_experiment.source_hash)
    used_hashes.add(experiment.source_hash)
    initial_experiment = experiment
    experiment_id = f"iteration-{iteration:03d}"
    artifact_dir = store.path / "checkpoints" / experiment_id
    initial_diff = _source_diff(parent_source, experiment.source, parent_id, experiment_id)
    atomic_write_text(store.path / "diffs" / f"{iteration:03d}-initial.diff", initial_diff)
    if proposed_experiment.source_hash != experiment.source_hash:
        atomic_write_text(
            store.path / "diffs" / f"{iteration:03d}-review.diff",
            _source_diff(
                proposed_experiment.source,
                experiment.source,
                f"{experiment_id}-proposed",
                f"{experiment_id}-reviewed",
            ),
        )
    experiment_path = Path("experiments") / f"{iteration:03d}.json"
    store.write_json(
        experiment_path,
        {
            "status": "selected",
            "parent_id": parent_id,
            "research_plan": research_plan.to_dict(),
            "research_responses": [_response_dict(response) for response in research_responses],
            "proposed_experiment": proposed_experiment.to_dict(),
            "experiment": experiment.to_dict(),
            "source_hash": experiment.source_hash,
            "responses": [_response_dict(response) for response in responses],
            "review": pre_execution_review.to_dict(),
            "review_responses": [_response_dict(response) for response in review_responses],
        },
    )

    started_at = utc_now()
    started = time.monotonic()
    recovery: list[dict[str, Any]] = []
    result: dict[str, Any] | None = None
    output = {"stdout": "", "stderr": ""}
    execution_error: str | None = None
    attempt = {
        "iteration": iteration,
        "experiment_id": experiment_id,
        "parent_id": parent_id,
        "hypothesis": experiment.hypothesis,
        "expected_effect": experiment.expected_effect,
        "research_plan": research_plan.to_dict(),
        "source_hash": experiment.source_hash,
        "source_hashes": sorted(used_hashes),
        "source_revision": expected_revision,
        "started_at": started_at,
        "recovery": recovery,
    }
    _begin_attempt(store, attempt)

    for repair_number in range(MAX_RUNTIME_REPAIRS + 1):
        store.event(
            "experiment_execution_started",
            iteration=iteration,
            experiment_id=experiment_id,
            source_hash=experiment.source_hash,
            repair_number=repair_number,
        )
        try:
            result, output = run_experiment(
                config,
                experiment,
                artifact_dir,
                research_data_path,
                timeout=_remaining_timeout(deadline),
                run_root=store.path,
                expected_source_revision=expected_revision,
            )
            execution_error = None
            if repair_number:
                recovery.append(
                    {
                        "experiment_id": experiment_id,
                        "source_hash": experiment.source_hash,
                        "action": "llm_code_repair_succeeded",
                    }
                )
            break
        except ExperimentFailure as error:
            execution_error = str(error)
            store.event("experiment_failed", iteration=iteration, error=execution_error)
            if repair_number >= MAX_RUNTIME_REPAIRS:
                recovery.append(
                    {
                        "experiment_id": experiment_id,
                        "source_hash": experiment.source_hash,
                        "error": execution_error,
                        "action": "repair_limit_reached_restore_previous_best",
                    }
                )
                break
            recovery.append(
                {
                    "experiment_id": experiment_id,
                    "source_hash": experiment.source_hash,
                    "error": execution_error,
                    "action": "request_llm_code_repair",
                }
            )
            attempt.update(recovery=recovery)
            _update_attempt(store, attempt)
        except HostRevisionChanged as error:
            state["status"] = "failed"
            state["stopping_reason"] = "host_revision_changed"
            state["failure"] = str(error)
            store.event("host_revision_changed", iteration=iteration, error=str(error))
            _persist_state(store, state, wall_started)
            raise

        try:
            store.event(
                "repair_requested",
                iteration=iteration,
                experiment_id=experiment_id,
                source_hash=experiment.source_hash,
                repair_number=repair_number + 1,
            )
            repaired, repair_responses = llm.repair(
                context,
                experiment,
                execution_error,
                plan=research_plan,
                deadline=deadline,
            )
            responses.extend(repair_responses)
            _add_usage(state, repair_responses)
            if repaired.source_hash in used_hashes:
                raise ValueError("repair repeated source that was already attempted")
            used_hashes.add(repaired.source_hash)
            next_repair = repair_number + 1
            repaired_id = f"iteration-{iteration:03d}-repair-{next_repair}"
            repair_diff = _source_diff(
                experiment.source,
                repaired.source,
                f"{experiment_id}-failed",
                repaired_id,
            )
            atomic_write_text(
                store.path / "diffs" / f"{iteration:03d}-repair-{next_repair}.diff",
                repair_diff,
            )
            experiment = repaired
            experiment_id = repaired_id
            artifact_dir = store.path / "checkpoints" / experiment_id
            attempt.update(
                experiment_id=experiment_id,
                hypothesis=experiment.hypothesis,
                expected_effect=experiment.expected_effect,
                source_hash=experiment.source_hash,
                source_hashes=sorted(used_hashes),
                recovery=recovery,
            )
            _update_attempt(store, attempt)
        except (ConnectionError, ValueError, LLMDeadlineExceeded) as repair_error:
            if isinstance(repair_error, GenerationFailure):
                responses.extend(repair_error.responses)
                _add_usage(state, repair_error.responses)
            elif isinstance(repair_error, LLMTransportFailure):
                responses.extend(repair_error.responses)
                _add_usage(state, repair_error.responses)
            execution_error = str(repair_error)
            recovery.append(
                {
                    "error": execution_error,
                    "action": "llm_code_repair_failed_restore_previous_best",
                }
            )
            store.event("repair_failed", iteration=iteration, error=execution_error)
            if isinstance(repair_error, LLMDeadlineExceeded):
                state["stopping_reason"] = "wall_clock_limit"
            break

    duration = time.monotonic() - started
    code_diff = _source_diff(parent_source, experiment.source, parent_id, experiment_id)
    code_diff_path = store.path / "diffs" / f"{iteration:03d}.diff"
    atomic_write_text(code_diff_path, code_diff)
    store.write_json(
        experiment_path,
        {
            "status": "executed" if result is not None else "failed",
            "parent_id": parent_id,
            "research_plan": research_plan.to_dict(),
            "research_responses": [_response_dict(response) for response in research_responses],
            "proposed_experiment": proposed_experiment.to_dict(),
            "pre_execution_review": pre_execution_review.to_dict(),
            "initial_experiment": initial_experiment.to_dict(),
            "initial_source_hash": initial_experiment.source_hash,
            "executed_experiment": experiment.to_dict(),
            "source_hash": experiment.source_hash,
            "responses": [_response_dict(response) for response in responses],
            "review_responses": [_response_dict(response) for response in review_responses],
            "recovery": recovery,
        },
    )

    record: dict[str, Any] = {
        "iteration": iteration,
        "id": experiment_id,
        "parent_id": parent_id,
        "hypothesis": experiment.hypothesis,
        "expected_effect": experiment.expected_effect,
        "research_plan": research_plan.to_dict(),
        "source_hash": experiment.source_hash,
        "initial_source_hash": initial_experiment.source_hash,
        "proposed_source_hash": proposed_experiment.source_hash,
        "pre_execution_review": pre_execution_review.to_dict(),
        "code_diff": code_diff,
        "code_diff_path": store.relative_path(code_diff_path),
        "source_revision": source_revision(),
        "seed": config.seed,
        "started_at": started_at,
        "duration_seconds": duration,
        "recovery": recovery,
        "stdout_tail": output["stdout"][-2000:],
        "stderr_tail": output["stderr"][-2000:],
        "manual_interventions": state["manual_interventions"],
    }
    if result is None:
        record.update(status="failed", error=execution_error, became_best=False)
    else:
        metrics = result["metrics"]
        is_best, significant = tracker.observe(float(metrics["primary"]))
        record.update(
            status="completed",
            metrics=metrics,
            diagnostics=_validation_diagnostics(
                metrics,
                state["baseline_valid"],
                best_before_experiment,
                history,
            ),
            epochs_completed=int(result.get("epochs_completed", 0)),
            pairs=int(result.get("pairs", 0)),
            checkpoint=result["checkpoint"],
            metadata=result.get("metadata", {}),
            became_best=is_best,
            significant_improvement=significant,
        )
        if is_best:
            state["best"] = {
                "id": experiment_id,
                "metrics": metrics,
                "checkpoint": result["checkpoint"],
                "experiment": {**experiment.to_dict(), "source_hash": experiment.source_hash},
            }

    # A runtime failure has already received one focused repair request; reflection is useful
    # only once there is a measured validation result to interpret.
    if result is not None:
        reflection = _reflect(llm, state, record, store, best_before_experiment, deadline)
        if reflection is not None:
            record["reflection"] = reflection.content
            record["reflection_response"] = _response_dict(reflection)
            _add_usage(state, [reflection])
    state["iteration"] = iteration
    state["stagnant"] = tracker.stagnant
    state["used_source_hashes"] = sorted(used_hashes)
    _commit_iteration(store, state, record, wall_started)
    _finish_attempt(store, iteration)


def _best_source(store: RunStore, state: dict[str, Any]) -> str:
    checkpoint = store.resolve_path(state["best"]["checkpoint"])
    source_path = checkpoint / "experiment.py" if checkpoint.is_dir() else None
    if source_path is None or not source_path.is_file():
        return ""
    return source_path.read_text(encoding="utf-8")


def _source_diff(before: str, after: str, before_id: str, after_id: str) -> str:
    lines = difflib.unified_diff(
        before.splitlines(),
        after.splitlines(),
        fromfile=f"{before_id}/experiment.py",
        tofile=f"{after_id}/experiment.py",
        lineterm="",
    )
    rendered = "\n".join(lines)
    return rendered + ("\n" if rendered else "")


def _generation_failure_record(
    iteration: int,
    parent_id: str,
    stage: str,
    error: Exception,
    research_plan: ResearchPlan | None,
    research_responses: Sequence[LLMResult],
    responses: Sequence[LLMResult],
    review_responses: Sequence[LLMResult],
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
        "research_responses": [_response_dict(response) for response in research_responses],
        "responses": [_response_dict(response) for response in responses],
        "review_responses": [_response_dict(response) for response in review_responses],
        "manual_interventions": state["manual_interventions"],
    }


def _reflect(
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
                    "code_diff": _bounded_context_text(
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


def _validation_diagnostics(
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
        "primary_delta_from_previous_best": float(
            metrics["primary"] - best_before["primary"]
        ),
        "matching_prior_iteration": matching_iteration,
    }


def _bounded_context_text(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    marker = "\n... <truncated for reflection> ...\n"
    prefix = (limit - len(marker)) // 2
    suffix = limit - len(marker) - prefix
    return value[:prefix] + marker + value[-suffix:]


def _response_dict(response: LLMResult) -> dict[str, Any]:
    return {
        "content": response.content,
        "usage": response.usage,
        "requested_model": response.requested_model,
        "returned_model": response.returned_model,
        "response_id": response.response_id,
        "finish_reason": response.finish_reason,
    }


def _add_usage(state: dict[str, Any], responses: Sequence[LLMResult]) -> None:
    for response in responses:
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            state["llm_usage"][key] += int(response.usage.get(key, 0))


def _remaining_timeout(deadline: float) -> int:
    remaining = int(deadline - time.monotonic())
    if remaining <= 0:
        raise LLMDeadlineExceeded("run wall-clock limit reached")
    return remaining
