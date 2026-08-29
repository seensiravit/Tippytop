"""Autonomous run lifecycle: preflight, baseline, search, and finalization."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from .artifacts import RunStore, assert_source_revision, read_json, source_revision, utc_now
from .config import RunConfig
from .doctor import run_doctor
from .research.context import random_sanity, snapshot_prior_research, summarize_dataset
from .runtime import prepare_research_data, run_reference_baseline
from .search import run_search
from .starter import load_splits
from .submission import finalize_run


RANDOM_REFERENCES = {
    "valid": {"GAUC": 0.4993, "nDCG@5": 0.4675, "primary": 0.4834},
    "test": {"GAUC": 0.4996, "nDCG@5": 0.4511, "primary": 0.4753},
}
FM_VALID_REFERENCE = 0.6016


def run_agent(
    config: RunConfig,
    *,
    run_dir: Path | None = None,
    finalize: bool = True,
) -> RunStore:
    config.validate()
    store, state = _open_run(config, run_dir)
    if state.get("status") == "completed":
        return store
    assert_source_revision(str(state["source_revision"]))

    wall_started = time.monotonic() - float(state.get("elapsed_seconds", 0.0))
    _run_preflight(config, store, state, wall_started)

    splits = load_splits(config.data_dir)
    _validate_harness(store, state, splits, wall_started)

    # Generated code receives this sanitized artifact, never the raw data directory or test rows.
    store.write_json("dataset_summary.json", summarize_dataset(splits))
    research_data_path = prepare_research_data(
        store.path / "research_data.pkl",
        config.data_dir,
        summary_path=store.path / "research_schema.json",
    )
    del splits
    _reproduce_baseline(config, store, state, wall_started)

    run_search(config, store, state, research_data_path, wall_started)
    if finalize:
        finalize_run(store, config, state)
    return store


def _open_run(config: RunConfig, run_dir: Path | None) -> tuple[RunStore, dict[str, Any]]:
    if run_dir is None:
        store = RunStore.create(config)
        state = _new_state(config)
        if config.prior_run is not None:
            memory = snapshot_prior_research(store, config.prior_run)
            state["used_source_hashes"] = memory["used_source_hashes"]
            state["research_parent"] = memory["source_run"]
            store.event(
                "prior_research_loaded",
                source_run=memory["source_run"],
                experiments=len(memory["recent_experiments"]),
                code_attempts=len(memory["recent_code_attempts"]),
            )
        store.write_json("state.json", state)
        return store, state

    store = RunStore.open(run_dir, config)
    state = read_json(store.path / "state.json")
    if state.get("status") != "completed":
        # Refuse a changed runtime before resume metadata mutates the durable run state.
        assert_source_revision(str(state["source_revision"]))
    active_config = config.to_dict(redact=True)
    if state.get("config") != active_config:
        previous_config = state.get("config")
        state["config"] = active_config
        store.write_json("config.json", active_config)
        store.write_json("state.json", state)
        store.event(
            "run_configuration_updated",
            previous=previous_config,
            active=active_config,
        )
    if state.get("status") != "completed":
        state["manual_interventions"] = int(state.get("manual_interventions", 0)) + 1
        store.write_json("state.json", state)
        store.event("run_resumed", iteration=state.get("iteration", 0))
    return store, state


def _run_preflight(
    config: RunConfig,
    store: RunStore,
    state: dict[str, Any],
    wall_started: float,
) -> None:
    if state.get("preflight_complete"):
        return
    checks = run_doctor(config, check_llm=not config.offline)
    store.write_json("doctor.json", checks)
    if not checks["ok"]:
        _fail(store, state, "preflight failed")
        raise RuntimeError("preflight failed: " + "; ".join(checks["errors"]))
    state["preflight_complete"] = True
    store.event("preflight_complete")
    _persist_state(store, state, wall_started)


def _validate_harness(
    store: RunStore,
    state: dict[str, Any],
    splits: dict[str, list[tuple[Any, ...]]],
    wall_started: float,
) -> None:
    if state.get("harness_complete"):
        return
    sanity = random_sanity(splits)
    store.write_json("random_sanity.json", sanity)
    deviations = {
        split: abs(sanity[split]["primary"] - RANDOM_REFERENCES[split]["primary"])
        for split in ("valid", "test")
    }
    if any(value > 0.001 for value in deviations.values()):
        message = f"random sanity mismatch: {deviations}"
        _fail(store, state, message)
        raise RuntimeError(message)
    state["harness_complete"] = True
    store.event("random_sanity_complete", metrics=sanity)
    _persist_state(store, state, wall_started)


def _reproduce_baseline(
    config: RunConfig,
    store: RunStore,
    state: dict[str, Any],
    wall_started: float,
) -> None:
    if state.get("baseline_complete"):
        return
    baseline_dir = store.path / "checkpoints" / "baseline-fm"
    result, output = run_reference_baseline(config, baseline_dir)
    store.write_json("baseline_output.json", output)
    metrics = result["metrics"]
    if abs(float(metrics["primary"]) - FM_VALID_REFERENCE) > 0.003:
        message = (
            f"FM validation primary {metrics['primary']:.4f} is not within 0.003 "
            f"of {FM_VALID_REFERENCE:.4f}"
        )
        _fail(store, state, message)
        raise RuntimeError(message)
    state["baseline_complete"] = True
    state["baseline_valid"] = metrics
    state["best"] = {
        "id": "baseline-fm",
        "metrics": metrics,
        "checkpoint": store.relative_path(Path(result["checkpoint"])),
        "experiment": {
            "kind": "trusted_reference",
            "hypothesis": "Reproduce the organizer pointwise FM baseline.",
            "expected_effect": "Establish a trustworthy validation reference.",
            "source_hash": None,
        },
    }
    store.event("baseline_complete", metrics=metrics)
    _persist_state(store, state, wall_started)


def _new_state(config: RunConfig) -> dict[str, Any]:
    return {
        "status": "running",
        "started_at": utc_now(),
        "source_revision": source_revision(),
        "iteration": 0,
        "stagnant": 0,
        "elapsed_seconds": 0.0,
        "manual_interventions": 0,
        "llm_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "used_source_hashes": [],
        "test_evaluated": False,
        "config": config.to_dict(redact=True),
    }


def _persist_state(store: RunStore, state: dict[str, Any], wall_started: float) -> None:
    state["elapsed_seconds"] = time.monotonic() - wall_started
    store.write_json("state.json", state)


def _fail(store: RunStore, state: dict[str, Any], message: str) -> None:
    state["status"] = "failed"
    state["failure"] = message
    store.write_json("state.json", state)
