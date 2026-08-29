"""One auditable scientist-coder-executor experiment iteration."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from ..artifacts import (
    RunStore,
    assert_source_revision,
    atomic_write_text,
    read_jsonl,
    source_revision,
    utc_now,
)
from ..config import RunConfig
from ..convergence import ConvergenceTracker
from ..llm import (
    GenerationFailure,
    LLMClient,
    LLMDeadlineExceeded,
    LLMResult,
    LLMTransportFailure,
)
from ..research import ResearchPlan
from ..research.context import build_research_context
from .execution import execute_with_repairs
from .journal import (
    begin_attempt as _begin_attempt,
    commit_iteration as _commit_iteration,
    finish_attempt as _finish_attempt,
    persist_state as _persist_state,
)
from .records import (
    add_usage as _add_usage,
    best_source as _best_source,
    generation_failure_record as _generation_failure_record,
    reflect as _reflect,
    response_dict as _response_dict,
    source_diff as _source_diff,
    validation_diagnostics as _validation_diagnostics,
)


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
    research_plan: ResearchPlan | None = None
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
            state,
        )
        state["iteration"] = iteration
        state["stagnant"] = tracker.stagnant
        _commit_iteration(store, state, record, wall_started)
        return

    used_hashes.add(experiment.source_hash)
    initial_experiment = experiment
    experiment_id = f"iteration-{iteration:03d}"
    initial_diff = _source_diff(parent_source, experiment.source, parent_id, experiment_id)
    atomic_write_text(store.path / "diffs" / f"{iteration:03d}-initial.diff", initial_diff)
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
        },
    )

    started_at = utc_now()
    recovery: list[dict[str, Any]] = []
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
        "experiment": experiment.to_dict(),
        "research_responses": [_response_dict(response) for response in research_responses],
        "responses": [_response_dict(response) for response in responses],
    }
    _begin_attempt(store, attempt)
    outcome = execute_with_repairs(
        config,
        store,
        state,
        llm,
        context,
        research_plan,
        experiment,
        experiment_id,
        used_hashes,
        responses,
        research_data_path,
        expected_revision,
        deadline,
        wall_started,
        attempt,
    )
    experiment = outcome.experiment
    experiment_id = outcome.experiment_id
    result = outcome.result
    output = outcome.output
    execution_error = outcome.error
    recovery = outcome.recovery
    duration = outcome.duration_seconds
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
            "initial_experiment": initial_experiment.to_dict(),
            "initial_source_hash": initial_experiment.source_hash,
            "executed_experiment": experiment.to_dict(),
            "source_hash": experiment.source_hash,
            "responses": [_response_dict(response) for response in responses],
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
