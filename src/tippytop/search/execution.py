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
from ..research import ResearchPlan
from ..runtime import ExperimentFailure, run_experiment
from .journal import persist_state, update_attempt
from .records import add_usage, source_diff


MAX_RUNTIME_REPAIRS = 2


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

    for repair_number in range(MAX_RUNTIME_REPAIRS + 1):
        store.event(
            "experiment_execution_started",
            iteration=attempt["iteration"],
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
            store.event(
                "experiment_failed",
                iteration=attempt["iteration"],
                error=execution_error,
            )
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

        try:
            store.event(
                "repair_requested",
                iteration=attempt["iteration"],
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
            add_usage(state, repair_responses)
            if repaired.source_hash in used_hashes:
                raise ValueError("repair repeated source that was already attempted")
            used_hashes.add(repaired.source_hash)
            next_repair = repair_number + 1
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
            attempt.update(
                experiment_id=experiment_id,
                hypothesis=experiment.hypothesis,
                expected_effect=experiment.expected_effect,
                source_hash=experiment.source_hash,
                source_hashes=sorted(used_hashes),
                recovery=recovery,
            )
            update_attempt(store, attempt)
        except (ConnectionError, ValueError, LLMDeadlineExceeded) as repair_error:
            if isinstance(repair_error, (GenerationFailure, LLMTransportFailure)):
                responses.extend(repair_error.responses)
                add_usage(state, repair_error.responses)
            execution_error = str(repair_error)
            recovery.append(
                {
                    "error": execution_error,
                    "action": "llm_code_repair_failed_restore_previous_best",
                }
            )
            store.event(
                "repair_failed",
                iteration=attempt["iteration"],
                error=execution_error,
            )
            if isinstance(repair_error, LLMDeadlineExceeded):
                state["stopping_reason"] = "wall_clock_limit"
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
