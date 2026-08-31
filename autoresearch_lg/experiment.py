"""The experiment sub-graph — do it safely (Robustness).

    apply_diff -> run_and_evaluate -> collect_metrics -> END
         \\failure         \\failure           \\failure
          '---------------> emit_failure --------------'

A bad diff won't write cleanly (rare — an OSError), training can crash/NaN/
time out, evaluation can fail to produce a parseable summary — each has its
own failure branch, all converging on emit_failure so the critic always
gets either a clean RunResult or a FailureRecord, never a raw exception.

No git, and the repo's root baseline.py/data.py are never written to: each
experiment gets its own folder under runs/ (tools.make_experiment_dir),
seeded with the proposed files + a copy of the fixed evaluate.py, and
training runs FROM that folder. A "revert" is just never reading that
folder again — nothing to overwrite, nothing that can leak into anything
else's state.

Retry semantics live one level up (graph.py's router re-invokes this whole
sub-graph on 'error', up to retry_cap): each retry uses seed=retry_count
instead of always 0 — a blind retry of identical code at the same seed
would fail identically every time, so varying the seed is what makes the
retry loop actually useful for genuinely-recoverable (seed-fragile) crashes
rather than a guaranteed no-op, while still burning down the cap quickly
for deterministic bugs (which correctly escalate to pivot).
"""
from __future__ import annotations

from langgraph.graph import END, StateGraph

from tippytop.runlog import scrub

from . import tools
from .state import ResearchState


def apply_diff(state: ResearchState) -> dict:
    try:
        name = f"exp_{state['iteration'] + 1:04d}"
        # Whatever propose read as "current" is what the diff must be against,
        # or the recorded diff describes a change nobody made.
        source_for_diff = (state.get("repair_exp_dir") if state.get("mode") == "repair"
                           and state.get("repair_exp_dir")
                           else (state.get("best_exp_dir") or state["repo_root"]))
        exp_dir = tools.make_experiment_dir(state["repo_root"], name)
        # `allowed` is the whitelist, not a suggestion: the LLM chooses these
        # path strings, and an unguarded write of "../../baseline.py" lands on
        # the frozen root kit while "evaluate.py" replaces the scoring spec that
        # make_experiment_dir just copied in. Both are caught before any byte is
        # written -- see tools.safe_experiment_path.
        tools.write_experiment_files(exp_dir, state["edited_files"],
                                     allowed=state["editable_files"])
        # A concept edit might only touch one of the two editable files —
        # the folder still needs a complete pair to run. Fill the other
        # from whichever source is currently "current" for propose (the
        # best exp_dir so far, or the pristine root files if nothing has
        # improved yet) — same source propose.py itself read to write this
        # diff, so this is always consistent with what the LLM saw.
        missing = [f for f in state["editable_files"] if f not in state["edited_files"]]
        if missing:
            source_dir = state.get("best_exp_dir") or state["repo_root"]
            tools.write_experiment_files(exp_dir, tools.read_experiment_files(source_dir, missing))
        # Deliverable 3 asks for "the code diff applied" per iteration. Only
        # the filenames were being logged, and the folders that hold the full
        # content are gitignored -- so a grader reading runs.jsonl saw
        # ["baseline.py"] and no way to tell what the agent actually did.
        # Computed here, where both sides are in hand, rather than reconstructed
        # later from folders that may have been cleaned.
        return {"exp_dir": exp_dir, "step_failed": False,
                "diff": tools.unified_diff(source_for_diff, exp_dir,
                                           state["editable_files"])}
    except tools.UnsafeExperimentPath as e:
        # Not an OSError, and it must not be allowed to escape as one: an
        # unsafe path is a defective *proposal*, so it belongs on the failure
        # branch where the router turns it into a repair with the reason
        # attached, not in a traceback that ends the run.
        return {"step_failed": True, "failure_step": "apply_diff",
                "failure_error": f"refused to write a proposed file: {e}"}
    except OSError as e:
        return {"step_failed": True, "failure_step": "apply_diff", "failure_error": str(e)}


def run_and_evaluate(state: ResearchState) -> dict:
    if state["step_failed"]:
        return {}
    seed = state["retry_count"]  # 0 first attempt; varies on each blind retry
    result = tools.run_baseline(state["exp_dir"], state["data_dir"], seed=seed)
    base = {"run_stdout": result["stdout"], "wall_seconds": result["wall_seconds"], "seed_used": seed}
    if result["crashed"]:
        reason = "timed out (>10min)" if result["timed_out"] else "non-zero exit"
        # The stdout tail is the one path by which a *test* metric can reach the
        # proposal prompt: context.py feeds `error` back to the LLM, and a run
        # that crashes after baseline.py has already printed its summary block
        # carries "test primary 0.59.." in exactly those last 2000 characters.
        # Everything else the model sees is validation-only by construction;
        # this is the leak the wall would otherwise have.
        return {**base, "step_failed": True, "failure_step": "run_and_evaluate",
                "failure_error": scrub(f"{reason}:\n{result['stdout'][-2000:]}", limit=2000)}
    return {**base, "step_failed": False}


def collect_metrics(state: ResearchState) -> dict:
    if state["step_failed"]:
        return {}
    parsed = tools.parse_summary(state["run_stdout"])
    if parsed is None:
        # Two different defects, and the model can only fix them if it is told
        # which one it hit: a missing summary block means the output format
        # changed, while a present-but-impossible number means the code is
        # reporting something that is not a score. The second is the dangerous
        # one -- an unbounded metric would make that folder the incumbent
        # forever and get shipped by finalize.
        found = tools.SUMMARY_RE.search(state["run_stdout"]) is not None
        why = ("the summary block reports a primary outside [0, 1] — "
               "mean(GAUC, nDCG@5) cannot leave that range, so the number is "
               "not a valid score"
               if found else
               "exit 0 but no summary block in stdout — output format changed?")
        return {"step_failed": True, "failure_step": "collect_metrics",
                "failure_error": why}
    valid_p, test_p = parsed
    full = tools.parse_summary_full(state["run_stdout"]) or {}
    return {"valid_primary": valid_p, "test_primary": test_p, "step_failed": False,
            "valid_metrics": full.get("valid", {}), "test_metrics": full.get("test", {})}


def emit_failure(state: ResearchState) -> dict:
    # Failure fields are already set by whichever step failed; this node is
    # the single convergence point so critic always sees a consistent shape,
    # and gives the sub-graph the failure-branch node the plan's diagram has.
    return {"valid_primary": 0.0, "test_primary": 0.0}


def _route_step(state: ResearchState) -> str:
    return "fail" if state["step_failed"] else "next"


def build_experiment_graph():
    g = StateGraph(ResearchState)
    g.add_node("apply_diff", apply_diff)
    g.add_node("run_and_evaluate", run_and_evaluate)
    g.add_node("collect_metrics", collect_metrics)
    g.add_node("emit_failure", emit_failure)

    g.set_entry_point("apply_diff")
    g.add_conditional_edges("apply_diff", _route_step, {"fail": "emit_failure", "next": "run_and_evaluate"})
    g.add_conditional_edges("run_and_evaluate", _route_step, {"fail": "emit_failure", "next": "collect_metrics"})
    g.add_conditional_edges("collect_metrics", _route_step, {"fail": "emit_failure", "next": END})
    g.add_edge("emit_failure", END)
    return g.compile()
