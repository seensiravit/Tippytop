"""The main loop — thin control flow wiring the three sub-graphs together.

    START -> bootstrap -> eda -> propose -> experiment -> critic -> router
                             ^                                  |
                       |                    ,-------------+-------------.
                       |                    | error, retries left       |
                       |                    v                           |
                       |               [back to experiment]             |
                       |                                     other outcomes
                       |                                                v
                       |                                       check_convergence
                       `------------------ continue -------------------/ \\
                                                                          done
                                                                           v
                                                                       finalize -> END

`bootstrap` (new) fills in anything missing from the input using
bootstrap.default_state — whatever's actually on disk (repo root via git,
history/concepts/checkpoints, reconstructed counters) plus config defaults.
This is what lets LangGraph Studio work by hitting Submit on an empty `{}`
input, not just a fully hand-built state; `cli.py run` still passes its own
full overrides (from argparse), which just win outright since bootstrap
only fills gaps.

`eda`/`bootstrap` each run exactly once — the graph topology alone
guarantees this (nothing loops back to either), matching
technical-plan.md's "one-time exploratory analysis" framing for `eda`.

`router` and `check_convergence` are real nodes (not just edge functions) on
purpose: technical-plan.md's diagram treats them as two distinct decision
points ("the two routers"), and keeping them as separate, inspectable nodes
means LangGraph Studio shows that distinction instead of collapsing it into
one opaque branch.

propose/experiment/critic are themselves compiled StateGraphs (see
propose.py/experiment.py/critic.py) added as nodes here — each compiles and
is testable independently against a mock ResearchState, which is the whole
point of the module split (per technical-plan.md §9, three people build
these in parallel).

No git anywhere in this loop, and root baseline.py/data.py are never
written to: each experiment gets its own folder under runs/
(experiment.py's apply_diff), so critic's keep_or_revert does no file I/O
at all — "revert" just means the next propose call won't read that folder.
The only git this whole harness touches is `setup`'s branch creation
(cli.py) — see tools.py's experiment-folders section.
"""
from __future__ import annotations

import time
from pathlib import Path

from langgraph.graph import END, StateGraph

from . import bootstrap, tools
from .critic import build_critic_graph
from .experiment import build_experiment_graph
from .propose import build_propose_graph
from .state import ResearchState


def bootstrap_node(state: ResearchState) -> dict:
    return bootstrap.default_state(state)


def eda(state: ResearchState) -> dict:
    summary = tools.run_eda(state["repo_root"], state["data_dir"])
    return {"eda_summary": summary}


def _close_active_concept(state: ResearchState, reason: str) -> list:
    concepts = [dict(c) for c in state["concepts"]]
    active = next((c for c in concepts if c["id"] == state["active_concept_id"]), None)
    if active:
        active["status"] = "closed"
        active["closed_reason"] = reason
    return concepts


def router(state: ResearchState) -> dict:
    """The router: reads critic's outcome, sets the next mode (tune/expand/
    pivot), and — separately from *what* to try, which propose decides —
    closes out the active concept when the macro move away from it happens.
    Mirrors technical-plan.md §5's router() almost verbatim."""
    outcome = state["outcome"]

    if outcome == "error" and state["retry_count"] < state["retry_cap"]:
        return {"retry_count": state["retry_count"] + 1, "retry_now": True}

    updates: dict = {"retry_count": 0, "retry_now": False}
    if outcome == "error":
        # Retries exhausted on a persistently-broken concept — abandon it.
        concepts = _close_active_concept(
            state, f"pivoted (unrecoverable error after {state['retry_cap']} retries)")
        updates.update(mode="pivot", tune_count=0, concepts=concepts, active_concept_id="")
    elif outcome == "improved":
        if state["tune_count"] + 1 >= state["tune_cap"]:
            concepts = _close_active_concept(
                state, f"expanded (maxed out after {state['tune_cap']} tunes)")
            updates.update(mode="expand", tune_count=0, concepts=concepts, active_concept_id="")
        else:
            updates.update(mode="tune", tune_count=state["tune_count"] + 1)
    elif outcome == "parity":
        # Indistinguishable from the incumbent, which is NOT the same as worse.
        # A first implementation of a good concept usually lands here because it
        # is untuned -- spend a bounded number of tunes before giving up on it.
        # (See critic.classify_outcome: the agent's own first run proposed FFM,
        # hit parity, was called failed, and pivoted off the best direction we
        # have measured.) When the tune budget runs out the concept still never
        # delivered, so it PIVOTS rather than expands -- expand is for successes.
        if state["tune_count"] + 1 >= state["tune_cap"]:
            concepts = _close_active_concept(
                state, f"pivoted (parity after {state['tune_cap']} tunes)")
            updates.update(mode="pivot", tune_count=0, concepts=concepts,
                           active_concept_id="")
        else:
            updates.update(mode="tune", tune_count=state["tune_count"] + 1)
    else:  # "failed" — a clear regression below -epsilon
        concepts = _close_active_concept(state, "pivoted (no improvement)")
        updates.update(mode="pivot", tune_count=0, concepts=concepts, active_concept_id="")
    return updates


def _route_after_router(state: ResearchState) -> str:
    return "experiment" if state["retry_now"] else "check_convergence"


def check_convergence(state: ResearchState) -> dict:
    elapsed = time.time() - state["start_time"]
    plateau = state["no_improve_count"] >= state["n_plateau"]
    out_of_iters = state["iteration"] >= state["max_iterations"]
    out_of_time = elapsed >= state["max_wall_seconds"]
    return {"converged": bool(plateau or out_of_iters or out_of_time)}


def _route_after_convergence(state: ResearchState) -> str:
    return "finalize" if state["converged"] else "propose"


def finalize(state: ResearchState) -> dict:
    # No git anywhere in this loop: root baseline.py/data.py were never
    # written to (each experiment lived in its own runs/ folder), so there's
    # nothing to commit and nothing that needs reverting one last time.
    root = state["repo_root"]
    elapsed = time.time() - state["start_time"]

    report = {
        "best_valid_primary": state["best_valid_primary"],
        "best_test_primary": state["best_test_primary"],
        "best_checkpoint_id": state["best_checkpoint_id"],
        "best_exp_dir": state.get("best_exp_dir") or None,
        "iterations": state["iteration"],
        "elapsed_seconds": round(elapsed, 1),
        "tokens_in_total": sum(h.get("tokens_in", 0) for h in state["history"]),
        "tokens_out_total": sum(h.get("tokens_out", 0) for h in state["history"]),
        "concepts_tried": len(state["concepts"]),
        "concepts_confirmed": sum(
            1 for c in state["concepts"] if any(a["outcome"] == "improved" for a in c["attempts"])
        ),
    }
    submission_path = str(Path(root, "submission.csv"))
    if state.get("best_exp_dir"):
        ok, msg = tools.make_submission(state["best_exp_dir"], root, state["data_dir"], submission_path)
    else:
        # Nothing ever beat the baseline — submit the pristine root code as-is
        # (a plain read: submit.py --make trains a fresh in-memory copy, it
        # never writes to baseline.py).
        ok, msg = tools.make_submission(root, root, state["data_dir"], submission_path)
    report["submission"] = submission_path if ok else None
    report["submission_note"] = msg
    tools.save_resource_report(str(Path(root, "resource_report.json")), report)
    return {"resource_report": report, "converged": True}


def build_graph():
    g = StateGraph(ResearchState)
    g.add_node("bootstrap", bootstrap_node)
    g.add_node("eda", eda)
    g.add_node("propose", build_propose_graph())
    g.add_node("experiment", build_experiment_graph())
    g.add_node("critic", build_critic_graph())
    g.add_node("router", router)
    g.add_node("check_convergence", check_convergence)
    g.add_node("finalize", finalize)

    g.set_entry_point("bootstrap")
    g.add_edge("bootstrap", "eda")
    g.add_edge("eda", "propose")
    g.add_edge("propose", "experiment")
    g.add_edge("experiment", "critic")
    g.add_edge("critic", "router")
    g.add_conditional_edges(
        "router", _route_after_router,
        {"experiment": "experiment", "check_convergence": "check_convergence"},
    )
    g.add_conditional_edges(
        "check_convergence", _route_after_convergence,
        {"propose": "propose", "finalize": "finalize"},
    )
    g.add_edge("finalize", END)

    return g.compile()


# Module-level compiled instance, for LangGraph Studio / `langgraph dev`
# (langgraph.json points at this object directly — see its "graphs" entry).
graph = build_graph()
