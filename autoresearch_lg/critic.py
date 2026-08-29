"""The critic sub-graph — judgment + the required logs.

    compare_to_best -> keep_or_revert -> classify_outcome
    -> update_counters -> write_log -> END

write_log is the one place the AIDE-style record gets written — hypothesis,
concept, diff summary, metrics, outcome, error, tokens, wall-clock — in
JSONL (runs.jsonl, the required deliverable) plus results.tsv for backward
compatibility, AND the one place a new checkpoint index row gets saved.

keep_or_revert does no file I/O at all now: each experiment already lives
in its own folder (experiment.py's apply_diff), so "revert" just means
propose.py won't read this folder as `best_exp_dir` next time — nothing to
overwrite, root baseline.py/data.py were never touched in the first place.
Kept as its own node (rather than folded into classify_outcome) for
structural fidelity to technical-plan.md's sub-graph shape.
"""
from __future__ import annotations

import time

from langgraph.graph import END, StateGraph

from . import dashboard, tools
from .state import ResearchState

_OUTCOME_TO_STATUS = {"improved": "keep", "parity": "discard",
                      "failed": "discard", "error": "crash"}


def compare_to_best(state: ResearchState) -> dict:
    if state["step_failed"]:
        return {"delta": 0.0}
    return {"delta": state["valid_primary"] - state["best_valid_primary"]}


# An experiment that scores EXACTLY the incumbent, to the 4 decimals the summary
# prints, almost never means "this idea is precisely as good". It means the
# harness ran code the model did not change -- most often because the model added
# a new `--model` choice while `run_fm`, the only path the harness invokes, was
# left untouched. Observed twice on run3, costing two iterations and ~26k tokens
# of dead code. Surfaced to the model so it can correct itself rather than
# spending its budget tuning something that never runs.
_NO_OP_TOLERANCE = 1e-9


def detect_no_op(state: ResearchState) -> bool:
    """True when the result is bit-identical to the incumbent's."""
    return (not state["step_failed"]
            and abs(state.get("delta", 0.0)) < _NO_OP_TOLERANCE)


def keep_or_revert(state: ResearchState) -> dict:
    return {}


def classify_outcome(state: ResearchState) -> dict:
    """Three live outcomes, because "not better" and "worse" are not the same.

    The original two-way split sent everything below +epsilon to `failed`, which
    the router answers with `pivot`. That discards a whole class of information:
    a FIRST implementation of a genuinely good concept almost never clears
    +0.002 immediately, because it arrives untuned.

    Observed in this repo's own run log: the agent proposed field-aware FM,
    implemented it from scratch, scored valid 0.6015 against a 0.6015 incumbent,
    was classified `failed`, and pivoted away. FFM is the strongest model we have
    measured -- verified at 0.6025 +-0.0004 against FM's 0.6016 +-0.0003 over six
    seeds each. The agent had the right idea and the router threw it out.

    `parity` marks |delta| <= epsilon: statistically indistinguishable from the
    incumbent given a seed spread of ~0.0004-0.0008. That is evidence the concept
    is VIABLE BUT UNTUNED, not evidence it is wrong. The router answers it with
    `tune` (bounded by tune_cap), so one round of refinement is spent before the
    concept is abandoned. Only a clear regression below -epsilon is `failed`.
    """
    if state["step_failed"]:
        return {"outcome": "error"}
    eps = state["epsilon"]
    if state["delta"] > eps:
        return {"outcome": "improved"}
    if state["delta"] >= -eps:
        return {"outcome": "parity"}
    return {"outcome": "failed"}


def update_counters(state: ResearchState) -> dict:
    outcome = state["outcome"]
    updates: dict = {
        "iteration": state["iteration"] + 1,
        "no_improve_count": 0 if outcome == "improved" else state["no_improve_count"] + 1,
    }
    concepts = [dict(c) for c in state["concepts"]]
    active = next((c for c in concepts if c["id"] == state["active_concept_id"]), None)
    if active:
        active["attempts"] = active["attempts"] + [{
            "checkpoint_id": None,  # filled in by write_log once the checkpoint is saved
            "valid_primary": state["valid_primary"] if not state["step_failed"] else 0.0,
            "outcome": outcome,
        }]
    updates["concepts"] = concepts
    return updates


def write_log(state: ResearchState) -> dict:
    checkpoint_id = tools.save_checkpoint(
        state["store_path"], state["iteration"], state["active_concept_id"], state["exp_dir"],
        state.get("valid_primary", 0.0), state.get("test_primary", 0.0), state["outcome"],
    )

    concepts = [dict(c) for c in state["concepts"]]
    active = next((c for c in concepts if c["id"] == state["active_concept_id"]), None)
    if active and active["attempts"]:
        active["attempts"][-1]["checkpoint_id"] = checkpoint_id

    record = {
        "iteration": state["iteration"],
        "concept_id": state["active_concept_id"],
        "concept": state.get("idea_concept", ""),
        "hypothesis": state.get("idea_hypothesis", ""),
        "description": state.get("idea_description", ""),
        "files_changed": sorted(state.get("edited_files", {}).keys()),
        "checkpoint_id": checkpoint_id,
        "exp_dir": state["exp_dir"],
        "seed": state.get("seed_used", 0),
        "metrics": {
            "valid_primary": state.get("valid_primary", 0.0),
            "test_primary": state.get("test_primary", 0.0),
        },
        "outcome": state["outcome"],
        "mode": state["mode"],
        "error": state.get("failure_error") if state["step_failed"] else None,
        "tokens_in": state.get("tokens_in", 0),
        "tokens_out": state.get("tokens_out", 0),
        "wall_clock_s": state.get("wall_seconds", 0.0),
        "elapsed_total_s": time.time() - state["start_time"],
    }
    tools.append_jsonl(state["log_path"], record)
    tools.append_result(
        # "commit" column kept for backward compat with program.md's convention —
        # the value is now a checkpoint id, not a git SHA.
        state["results_path"], str(checkpoint_id), record["metrics"]["valid_primary"],
        record["metrics"]["test_primary"], record["wall_clock_s"],
        _OUTCOME_TO_STATUS[record["outcome"]], record["description"] or record["concept"],
    )
    tools.save_concepts(state["concepts_path"], concepts)
    dashboard.write_dashboard(
        state["repo_root"], state["results_path"], state["dashboard_path"], state["concepts_path"],
    )

    updates = {"history": state["history"] + [record], "concepts": concepts}
    if record["outcome"] == "improved":
        updates["best_checkpoint_id"] = checkpoint_id
        updates["best_exp_dir"] = state["exp_dir"]
        updates["best_valid_primary"] = record["metrics"]["valid_primary"]
        updates["best_test_primary"] = record["metrics"]["test_primary"]
    return updates


def build_critic_graph():
    g = StateGraph(ResearchState)
    g.add_node("compare_to_best", compare_to_best)
    g.add_node("keep_or_revert", keep_or_revert)
    g.add_node("classify_outcome", classify_outcome)
    g.add_node("update_counters", update_counters)
    g.add_node("write_log", write_log)

    g.set_entry_point("compare_to_best")
    g.add_edge("compare_to_best", "keep_or_revert")
    g.add_edge("keep_or_revert", "classify_outcome")
    g.add_edge("classify_outcome", "update_counters")
    g.add_edge("update_counters", "write_log")
    g.add_edge("write_log", END)
    return g.compile()
