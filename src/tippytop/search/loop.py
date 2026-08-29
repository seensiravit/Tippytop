"""Validation-only lifecycle for autonomous generated experiment search."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from ..artifacts import RunStore
from ..config import RunConfig
from ..convergence import ConvergenceTracker
from ..llm import LLMClient
from .iteration import run_iteration
from .journal import persist_state, recover_inflight_attempts, recover_transactions


def run_search(
    config: RunConfig,
    store: RunStore,
    state: dict[str, Any],
    research_data_path: Path,
    wall_started: float,
) -> None:
    recover_transactions(store, state)
    recover_inflight_attempts(store, state, wall_started)
    tracker = ConvergenceTracker(
        epsilon=config.epsilon,
        patience=config.patience,
        best=float(state["best"]["metrics"]["primary"]),
        stagnant=int(state.get("stagnant", 0)),
    )
    llm = None if config.offline else LLMClient(config)
    used_hashes = set(state.get("used_source_hashes", []))
    deadline = wall_started + config.max_hours * 3600

    if llm is None:
        state["stopping_reason"] = "offline_baseline_only"

    while llm is not None and int(state["iteration"]) < config.max_iterations:
        if tracker.converged:
            state["stopping_reason"] = "converged"
            break
        if time.monotonic() >= deadline:
            state["stopping_reason"] = "wall_clock_limit"
            break
        run_iteration(
            config,
            store,
            state,
            tracker,
            llm,
            used_hashes,
            research_data_path,
            wall_started,
            deadline,
        )

    if not state.get("stopping_reason"):
        state["stopping_reason"] = "iteration_limit"
    state["status"] = "selected"
    persist_state(store, state, wall_started)
    store.event(
        "search_complete",
        reason=state["stopping_reason"],
        best=state["best"]["metrics"],
    )
