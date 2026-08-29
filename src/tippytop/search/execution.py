"""Sandbox execution and bounded source-repair cycle for one experiment."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..artifacts import HostRevisionChanged, RunStore, atomic_write_text
from ..config import RunConfig
from ..generated import GeneratedExperiment
from ..llm import GenerationFailure, LLMClient, LLMDeadlineExceeded, LLMResult, LLMTransportFailure
from ..llm.api_validation import validate_installed_api_calls
from ..llm.semantic_validation import validate_prediction_paths
from ..research import ResearchPlan
from ..runtime import ExperimentFailure, run_experiment
from ..runtime.diagnostics import runtime_failure_diagnostics
from .journal import persist_state, update_attempt
from .records import add_usage, response_dict, source_diff


MAX_EXECUTED_REPAIRS = 8
MAX_REJECTED_REPAIR_REQUESTS_PER_FAILURE = 4


@dataclass(frozen=True)
class ExecutionOutcome:
    experiment: GeneratedExperiment
    experiment_id: str
    result: dict[str, Any] | None
    output: dict[str, str]
    error: str | None
    recovery: list[dict[str, Any]]
    duration_seconds: float


def execute_with_repairs(
    config: RunConfig,
    store: RunStore,
    state: dict[str, Any],
    llm: LLMClient,
    context: dict[str, Any],
    research_plan: ResearchPlan,
    experiment: GeneratedExperiment,
    experiment_id: str,
    used_hashes: set[str],
    responses: list[LLMResult],
    research_data_path: Path,
    expected_revision: str,
    deadline: float,
    wall_started: float,
    attempt: dict[str, Any],
) -> ExecutionOutcome:
    started = time.monotonic()
    recovery: list[dict[str, Any]] = attempt["recovery"]
    result: dict[str, Any] | None = None
    output = {"stdout": "", "stderr": ""}
    execution_error: str | None = None
    artifact_dir = store.path / "checkpoints" / experiment_id
    failure_history = _prior_failure_history(recovery)
    executed_repairs = 0
    repair_requests = 0

    while True:
        store.event(
            "experiment_execution_started",
            iteration=attempt["iteration"],
            experiment_id=experiment_id,
            source_hash=experiment.source_hash,
            repair_number=executed_repairs,
        )
        try:
            try:
                validate_installed_api_calls(experiment.source)
                validate_prediction_paths(experiment.source)
            except ValueError as error:
                raise ExperimentFailure(str(error)) from error
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
            if executed_repairs:
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
            failure = {
                "experiment_id": experiment_id,
                "source_hash": experiment.source_hash,
                "error": _bounded_error(execution_error),
                "diagnostics": runtime_failure_diagnostics(experiment.source, execution_error),
            }
            failure_history.append(failure)
            store.event(
                "experiment_failed",
                iteration=attempt["iteration"],
                error=execution_error,
            )
            if executed_repairs >= MAX_EXECUTED_REPAIRS:
                recovery.append(
                    {
                        "experiment_id": experiment_id,
                        "source_hash": experiment.source_hash,
                        "error": execution_error,
                        "diagnostics": failure["diagnostics"],
                        "action": "executed_repair_limit_reached_restore_previous_best",
                    }
                )
                attempt.update(recovery=recovery)
                update_attempt(store, attempt)
                break
            recovery.append(
                {
                    "experiment_id": experiment_id,
                    "source_hash": experiment.source_hash,
                    "error": execution_error,
                    "diagnostics": failure["diagnostics"],
                    "action": "request_llm_code_repair",
                }
            )
            attempt.update(recovery=recovery)
            update_attempt(store, attempt)
        except HostRevisionChanged as error:
            state["status"] = "failed"
            state["stopping_reason"] = "host_revision_changed"
            state["failure"] = str(error)
            store.event(
                "host_revision_changed",
                iteration=attempt["iteration"],
                error=str(error),
            )
            persist_state(store, state, wall_started)
            raise
        except LLMDeadlineExceeded as error:
            execution_error = str(error)
            recovery.append(
                {
                    "experiment_id": experiment_id,
                    "source_hash": experiment.source_hash,
                    "error": execution_error,
                    "action": "wall_clock_limit_restore_previous_best",
                }
            )
            attempt.update(recovery=recovery)
            update_attempt(store, attempt)
            state["stopping_reason"] = "wall_clock_limit"
            persist_state(store, state, wall_started)
            break

        repaired_ready = False
        rejected_requests = 0
        while rejected_requests < MAX_REJECTED_REPAIR_REQUESTS_PER_FAILURE:
            repair_requests += 1
            try:
                store.event(
                    "repair_requested",
                    iteration=attempt["iteration"],
                    experiment_id=experiment_id,
                    source_hash=experiment.source_hash,
                    repair_number=executed_repairs + 1,
                    request_number=repair_requests,
                )
                repaired, repair_responses = llm.repair(
                    context,
                    experiment,
                    execution_error,
                    plan=research_plan,
                    failure_history=failure_history,
                    deadline=deadline,
                )
                responses.extend(repair_responses)
                add_usage(state, repair_responses)
                if repaired.source_hash in used_hashes:
                    raise ValueError("repair repeated source that was already attempted")
                used_hashes.add(repaired.source_hash)
                next_repair = executed_repairs + 1
                repaired_id = f"iteration-{attempt['iteration']:03d}-repair-{next_repair}"
                atomic_write_text(
                    store.path / "diffs" / f"{attempt['iteration']:03d}-repair-{next_repair}.diff",
                    source_diff(
                        experiment.source,
                        repaired.source,
                        f"{experiment_id}-failed",
                        repaired_id,
                    ),
                )
                experiment = repaired
                experiment_id = repaired_id
                artifact_dir = store.path / "checkpoints" / experiment_id
                executed_repairs = next_repair
                attempt.update(
                    experiment_id=experiment_id,
                    hypothesis=experiment.hypothesis,
                    expected_effect=experiment.expected_effect,
                    source_hash=experiment.source_hash,
                    source_hashes=sorted(used_hashes),
                    recovery=recovery,
                    experiment=experiment.to_dict(),
                    responses=[response_dict(response) for response in responses],
                )
                update_attempt(store, attempt)
                repaired_ready = True
                break
            except LLMDeadlineExceeded as repair_error:
                execution_error = str(repair_error)
                recovery.append(
                    {
                        "error": execution_error,
                        "action": "wall_clock_limit_restore_previous_best",
                    }
                )
                state["stopping_reason"] = "wall_clock_limit"
                store.event(
                    "repair_failed",
                    iteration=attempt["iteration"],
                    error=execution_error,
                )
                attempt.update(recovery=recovery)
                update_attempt(store, attempt)
                break
            except (ConnectionError, ValueError) as repair_error:
                rejected_requests += 1
                if isinstance(repair_error, (GenerationFailure, LLMTransportFailure)):
                    responses.extend(repair_error.responses)
                    add_usage(state, repair_error.responses)
                execution_error = str(repair_error)
                rejected = {
                    "experiment_id": experiment_id,
                    "source_hash": experiment.source_hash,
                    "error": _bounded_error(execution_error),
                    "diagnostics": {},
                }
                failure_history.append(rejected)
                recovery.append(
                    {
                        **rejected,
                        "action": "llm_code_repair_response_rejected",
                    }
                )
                store.event(
                    "repair_response_rejected",
                    iteration=attempt["iteration"],
                    repair_number=executed_repairs + 1,
                    request_number=repair_requests,
                    error=execution_error,
                )
                attempt.update(
                    recovery=recovery,
                    responses=[response_dict(response) for response in responses],
                )
                update_attempt(store, attempt)

        if repaired_ready:
            continue
        if state.get("stopping_reason") != "wall_clock_limit":
            recovery.append(
                {
                    "experiment_id": experiment_id,
                    "source_hash": experiment.source_hash,
                    "error": execution_error,
                    "action": "repair_response_limit_reached_restore_previous_best",
                }
            )
            store.event(
                "repair_failed",
                iteration=attempt["iteration"],
                error=execution_error,
            )
            attempt.update(
                recovery=recovery,
                responses=[response_dict(response) for response in responses],
            )
            update_attempt(store, attempt)
        break

    return ExecutionOutcome(
        experiment=experiment,
        experiment_id=experiment_id,
        result=result,
        output=output,
        error=execution_error,
        recovery=recovery,
        duration_seconds=time.monotonic() - started,
    )


def _remaining_timeout(deadline: float) -> int:
    remaining = int(deadline - time.monotonic())
    if remaining <= 0:
        raise LLMDeadlineExceeded("run wall-clock limit reached")
    return remaining


def _prior_failure_history(recovery: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "experiment_id": item.get("experiment_id"),
            "source_hash": item.get("source_hash"),
            "error": _bounded_error(str(item["error"])),
            "diagnostics": item.get("diagnostics", {}),
        }
        for item in recovery
        if item.get("error") and item.get("source_hash")
    ]


def _bounded_error(error: str, limit: int = 1200) -> str:
    if len(error) <= limit:
        return error
    prefix = error[:300]
    suffix = error[-(limit - len(prefix) - 40) :]
    return f"{prefix}\n... traceback truncated ...\n{suffix}"
