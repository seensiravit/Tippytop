"""Crash-consistent state and iteration-log persistence for autonomous search."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from ..artifacts import RunStore, atomic_write_jsonl, read_json, read_jsonl, utc_now


def persist_state(store: RunStore, state: dict[str, Any], wall_started: float) -> None:
    state["elapsed_seconds"] = time.monotonic() - wall_started
    store.write_json("state.json", state)


def commit_iteration(
    store: RunStore,
    state: dict[str, Any],
    record: dict[str, Any],
    wall_started: float,
) -> None:
    """Commit state and audit record through one recoverable transaction file."""

    state["elapsed_seconds"] = time.monotonic() - wall_started
    iteration = int(record["iteration"])
    store.write_json(
        Path("transactions") / f"{iteration:03d}.json",
        {"iteration": iteration, "record": record, "state": state},
    )
    store.write_json("state.json", state)
    _rebuild_iteration_log(store)


def recover_transactions(store: RunStore, state: dict[str, Any]) -> None:
    transaction_dir = store.path / "transactions"
    if not transaction_dir.is_dir():
        transaction_dir.mkdir(parents=True)
        return
    transactions = [
        read_json(path)
        for path in sorted(transaction_dir.glob("[0-9][0-9][0-9].json"))
    ]
    if transactions:
        latest = max(transactions, key=lambda item: int(item["iteration"]))
        if int(latest["iteration"]) > int(state.get("iteration", 0)):
            state.clear()
            state.update(latest["state"])
            store.write_json("state.json", state)
        _rebuild_iteration_log(store)


def begin_attempt(store: RunStore, attempt: dict[str, Any]) -> None:
    """Persist an iteration before generated code starts consuming resources."""

    iteration = int(attempt["iteration"])
    store.write_json(
        Path("transactions") / f"{iteration:03d}.inflight.json",
        {**attempt, "status": "in_progress", "updated_at": utc_now()},
    )


def update_attempt(store: RunStore, attempt: dict[str, Any]) -> None:
    begin_attempt(store, attempt)


def finish_attempt(store: RunStore, iteration: int) -> None:
    (store.path / "transactions" / f"{iteration:03d}.inflight.json").unlink(missing_ok=True)


def recover_inflight_attempts(
    store: RunStore,
    state: dict[str, Any],
    wall_started: float,
) -> None:
    """Convert a killed in-progress attempt into one durable, non-measured iteration."""

    transaction_dir = store.path / "transactions"
    for path in sorted(transaction_dir.glob("*.inflight.json")):
        attempt = read_json(path)
        iteration = int(attempt["iteration"])
        completed = transaction_dir / f"{iteration:03d}.json"
        if completed.is_file() or iteration <= int(state.get("iteration", 0)):
            path.unlink(missing_ok=True)
            continue
        hashes = set(state.get("used_source_hashes", []))
        hashes.update(str(value) for value in attempt.get("source_hashes", []) if value)
        state["used_source_hashes"] = sorted(hashes)
        state["iteration"] = iteration
        record = {
            "iteration": iteration,
            "id": attempt["experiment_id"],
            "parent_id": attempt["parent_id"],
            "status": "interrupted",
            "error": "the previous process stopped during this generated experiment",
            "hypothesis": attempt["hypothesis"],
            "expected_effect": attempt["expected_effect"],
            "source_hash": attempt["source_hash"],
            "source_revision": attempt["source_revision"],
            "started_at": attempt["started_at"],
            "became_best": False,
            "manual_interventions": state.get("manual_interventions", 0),
            "recovery": attempt.get("recovery", []),
            "research_plan": attempt.get("research_plan"),
            "research_responses": attempt.get("research_responses", []),
            "responses": attempt.get("responses", []),
            "executed_experiment": attempt.get("experiment"),
        }
        commit_iteration(store, state, record, wall_started)
        path.unlink(missing_ok=True)
        store.event("interrupted_attempt_recovered", iteration=iteration)


def _rebuild_iteration_log(store: RunStore) -> None:
    records = {
        int(record["iteration"]): record
        for record in read_jsonl(store.path / "iterations.jsonl")
    }
    transaction_dir = store.path / "transactions"
    for path in sorted(transaction_dir.glob("[0-9][0-9][0-9].json")):
        transaction = read_json(path)
        records[int(transaction["iteration"])] = transaction["record"]
    atomic_write_jsonl(
        store.path / "iterations.jsonl",
        [records[iteration] for iteration in sorted(records)],
    )
