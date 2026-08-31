"""CLI driver for the LangGraph autoresearch harness.

    python -m autoresearch_lg.cli setup     --tag aug29
    python -m autoresearch_lg.cli run       --tag aug29
    python -m autoresearch_lg.cli dashboard
    python -m autoresearch_lg.cli graph     [--out graph.mmd]

`setup` mirrors technical-plan.md's "reproduce the baseline" gate: branch,
results.tsv/runs.jsonl/concepts.json/checkpoints.db, first baseline run
logged as the starting point (not a proposed concept — nothing to tune/
expand/pivot yet; runs directly against the pristine root baseline.py,
the one and only time this harness executes the root files in place).

`run` no longer loops in Python — the loop lives in the graph itself
(propose -> experiment -> critic -> router -> check_convergence, cycling
until converged), including `finalize` (submission.csv + resource_report.json)
firing automatically once convergence trips. `run` makes ONE `.stream()`
call over the whole thing and is safe to Ctrl+C at any point: every
completed experiment is already durable on disk (results.tsv/runs.jsonl/
concepts.json/checkpoints.db) by the time control returns here.

No git commits or resets anywhere in the loop, and root baseline.py/data.py
are never written to once `setup` finishes: every experiment gets its own
folder under runs/ (see experiment.py), so there's nothing to revert and
nothing that can be left in a half-modified state.

Requires an API key for whichever provider `--model` names — ANTHROPIC_API_KEY
for claude-* models (default), OPENAI_API_KEY for anything else (propose.py
dispatches on the model name prefix). Only `dashboard` and `graph` run
without one.
"""
from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

from dotenv import load_dotenv

from tippytop.runlog import InterventionLog

from . import bootstrap
from . import dashboard as dashboard_mod
from . import resilience
from . import tools

load_dotenv()  # picks up ANTHROPIC_API_KEY / OPENAI_API_KEY from .env in the cwd, if present

# Re-exported for backward compat (other modules/tests importing these from
# cli.py); the actual definitions live in bootstrap.py now, shared with
# graph.py's `bootstrap` node.
DATA_FILES = bootstrap.DATA_FILES
repo_root = bootstrap.repo_root
build_system_prompt = bootstrap.build_system_prompt
check_data_dir = bootstrap.check_data_dir


# --------------------------------------------------------------- setup ----
def cmd_setup(args: argparse.Namespace) -> None:
    root = repo_root()
    branch = tools.create_experiment_branch(root, args.tag)
    print(f"created branch {branch}")

    if not check_data_dir(args.data_dir):
        print(
            f"ERROR: {args.data_dir} is missing one or more of the six "
            "KuaiRand-Pure CSVs. Follow the download steps in README.md, "
            "then re-run setup."
        )
        sys.exit(1)

    # A new tag means a new run. Run artifacts live at the repo root, not under
    # the tag, and bootstrap derives `iteration` from the length of runs.jsonl —
    # so without this the new run inherits the previous one's iteration count
    # and can hit its budget before running anything.
    archived = tools.archive_run_artifacts(root)
    if archived:
        print(f"archived the previous run's artifacts to {archived}")

    results_path = str(Path(root, "results.tsv"))
    dashboard_path = str(Path(root, "results_dashboard.html"))
    concepts_path = str(Path(root, "concepts.json"))
    log_path = str(Path(root, "runs.jsonl"))
    store_path = str(Path(root, "checkpoints.db"))
    tools.init_results_tsv(results_path)
    tools.save_concepts(concepts_path, [])
    tools.init_store(store_path)
    bootstrap.start_time_path(root).write_text(str(time.time()), encoding="utf-8")

    print("running baseline (FM, seed=0) to record the starting point...")
    result = tools.run_baseline(root, args.data_dir, seed=0)
    parsed = None if result["crashed"] else tools.parse_summary(result["stdout"])
    if parsed is None:
        print("baseline run crashed or produced no summary — aborting setup.")
        print(result["stdout"][-2000:])
        sys.exit(1)
    valid_p, test_p = parsed

    # No runs/ folder for the baseline itself — "" exp_dir means propose.py's
    # first call reads root's pristine files directly, which is exactly what
    # should happen for the very first proposal.
    checkpoint_id = tools.save_checkpoint(store_path, 0, "", "", valid_p, test_p, "improved")

    tools.append_result(
        results_path, str(checkpoint_id), valid_p, test_p, result["wall_seconds"],
        "keep", "baseline (FM, k=16)",
    )
    tools.append_jsonl(log_path, {
        "iteration": 0, "concept_id": "", "concept": "baseline",
        "hypothesis": "", "description": "baseline (FM, k=16)",
        "files_changed": [], "checkpoint_id": checkpoint_id, "exp_dir": "", "seed": 0,
        "metrics": {"valid_primary": valid_p, "test_primary": test_p},
        "outcome": "improved", "mode": "baseline", "error": None,
        "tokens_in": 0, "tokens_out": 0,
        "wall_clock_s": result["wall_seconds"], "elapsed_total_s": result["wall_seconds"],
    })
    dashboard_mod.write_dashboard(root, results_path, dashboard_path, concepts_path)

    print(f"baseline logged: valid_primary={valid_p:.4f} test_primary={test_p:.4f} "
          f"(checkpoint {checkpoint_id})")
    print(f"dashboard written to {dashboard_path}")
    print(f"\nnext: python -m autoresearch_lg.cli run --tag {args.tag}")


# ----------------------------------------------------------------- run ----
def _load_state(root: str, args: argparse.Namespace) -> dict:
    """Thin wrapper: turn argparse args into overrides and hand off to
    bootstrap.default_state (the shared logic graph.py's `bootstrap` node
    also uses)."""
    return bootstrap.default_state({
        "repo_root": root,
        "data_dir": args.data_dir,
        "model": args.model,
        "max_iterations": args.max_iterations,
        "max_wall_seconds": args.max_wall_hours * 3600,
        "epsilon": args.epsilon,
        "n_plateau": args.n_plateau,
        "retry_cap": args.retry_cap,
        "tune_cap": args.tune_cap,
    })


def cmd_run(args: argparse.Namespace) -> None:
    root = repo_root()
    branch = tools.current_branch(root)
    expected = f"autoresearch/{args.tag}"
    if branch != expected:
        print(f"WARNING: current branch is '{branch}', expected '{expected}'.")
        if input("continue anyway? [y/N] ").strip().lower() != "y":
            sys.exit(1)

    # Autonomy is graded on the manual-intervention count, so a restart has to
    # cost the run something. This is detected, not declared: if runs.jsonl
    # already holds iterations, the operator has restarted a loop that was
    # supposed to finish on its own, whether or not they choose to say so.
    ilog = InterventionLog(Path(root))
    rec = ilog.detect_resume(Path(root, "runs.jsonl"))
    if rec is not None:
        print(f"intervention recorded: {rec.reason} (total {ilog.count})")

    from . import graph as graph_mod  # deferred: needs langgraph + anthropic installed
    compiled = graph_mod.build_graph()

    state = _load_state(root, args)
    print(f"starting from best valid_primary={state['best_valid_primary']:.4f}, "
          f"{state['iteration']}/{state['max_iterations']} iterations used, "
          f"budget {args.max_wall_hours:.1f}h.")

    recursion_limit = (state["max_iterations"] - state["iteration"] + 5) * 25 + 20
    last_history_len = state["iteration"]
    final_state = state
    stop_reason = "converged"
    try:
        for chunk in compiled.stream(state, config={"recursion_limit": recursion_limit}, stream_mode="values"):
            final_state = chunk
            if len(chunk["history"]) > last_history_len:
                last_history_len = len(chunk["history"])
                last = chunk["history"][-1]
                print(f"\n--- experiment {last_history_len} "
                      f"[{last['mode']}/{last['outcome']}] {last['concept_id']} ---")
                print(f"concept: {last['concept']}")
                print(f"idea: {last['description']}")
                print(f"result: valid={last['metrics']['valid_primary']:.4f} "
                      f"test={last['metrics']['test_primary']:.4f} "
                      f"wall={last['wall_clock_s']:.1f}s")
                if last.get("error"):
                    print(f"error: {last['error'][:300]}")
    except KeyboardInterrupt:
        stop_reason = "interrupted (KeyboardInterrupt)"
        print("\ninterrupted — state on disk is consistent up to the last completed "
              "node (results.tsv/runs.jsonl/concepts.json/checkpoints.db). "
              "Re-run `run` to continue.")
    except Exception as e:                                     # noqa: BLE001
        # Nothing above should reach here: every expected failure has an edge,
        # transient provider errors retry, and a terminal one routes to
        # finalize. This catches the unexpected -- and the point is that even
        # an unforeseen crash still ships the deliverables below, instead of
        # throwing away every iteration the run completed.
        stop_reason = f"crashed ({type(e).__name__}: {e})"
        resilience.log_recovery(root, resilience.recovery_event(
            iteration=final_state.get("iteration", 0), layer="terminal",
            kind=type(e).__name__, action="finalize-early", detail=str(e)))
        print(f"\nunhandled error: {type(e).__name__}: {e}", file=sys.stderr)
        traceback.print_exc()
    finally:
        # The failsafe. Whatever happened above -- clean convergence, Ctrl+C, a
        # crash, the provider going away -- the graded artifacts get written
        # from what is on disk. A no-op if the graph's own finalize already ran.
        finalize_from_disk(root, args, reason=stop_reason)

    print(f"\nbest valid_primary={final_state['best_valid_primary']:.4f} "
          f"(checkpoint {final_state['best_checkpoint_id']}, "
          f"folder: {final_state.get('best_exp_dir') or '(baseline — no experiment beat it)'}). "
          f"dashboard: {final_state['dashboard_path']}")
    if final_state.get("converged") and final_state.get("resource_report"):
        r = final_state["resource_report"]
        print(f"\nconverged. {r['iterations']} iterations, {r['elapsed_seconds']:.0f}s elapsed, "
              f"{r['tokens_in_total']}+{r['tokens_out_total']} tokens in/out, "
              f"{r['concepts_tried']} concepts tried ({r['concepts_confirmed']} confirmed).")
        print(f"manual interventions: {r.get('intervention_summary', 'n/a')}")
        print(f"submission: {r['submission'] or 'FAILED — ' + r['submission_note']}")


# --------------------------------------------------------- finalize ------
def _already_finalized(root: str) -> bool:
    return Path(root, "resource_report.json").exists()


def finalize_from_disk(root: str, args: argparse.Namespace, *, reason: str) -> None:
    """Produce the graded deliverables from whatever is on disk, right now.

    This is the L4 failsafe, and it is the one that matters most. `finalize`
    used to be reachable only by falling off the end of the loop, so any exit
    that was not a clean convergence -- Ctrl+C, an unhandled exception, a
    machine going to sleep -- left `submission.csv` and `resource_report.json`
    unwritten. A run with 30 good iterations in `runs.jsonl` then scored zero on
    Deliverables 3 and 4, because a grader cannot mark files that do not exist.

    Safe to call twice: it is a no-op once the report exists, so it never
    re-trains a submission model or overwrites a good artifact with a worse one.
    """
    if _already_finalized(root):
        return
    from . import graph as graph_mod
    state = _load_state(root, args)
    state["stop_reason"] = reason
    print(f"\nfinalizing from disk ({reason}) — {state['iteration']} iterations on record...")
    try:
        out = graph_mod.finalize(state)
    except Exception as e:                                    # noqa: BLE001
        print(f"finalize failed: {type(e).__name__}: {e}", file=sys.stderr)
        print("runs.jsonl / results.tsv / concepts.json are still on disk and "
              "still gradeable; re-run `finalize` once the cause is fixed.",
              file=sys.stderr)
        return
    r = out["resource_report"]
    print(f"wrote resource_report.json — {r['iterations']} iterations, "
          f"{r['elapsed_seconds'] / 3600:.2f}h, "
          f"{r['tokens_in_total']}+{r['tokens_out_total']} tokens, "
          f"{r.get('recovery_events', 0)} recovery events, "
          f"{r.get('manual_interventions', 0)} manual interventions.")
    print(f"submission: {r['submission'] or 'FAILED — ' + str(r['submission_note'])}")


def cmd_finalize(args: argparse.Namespace) -> None:
    root = repo_root()
    if args.force and _already_finalized(root):
        Path(root, "resource_report.json").unlink()
    finalize_from_disk(root, args, reason=args.reason)


# ------------------------------------------------------------- note ------
def cmd_note(args: argparse.Namespace) -> None:
    """Record an intervention the harness cannot detect for itself.

    Resumes and hand-edited seeds are caught automatically. Everything else --
    editing source between runs, restarting after a manual dependency install,
    steering the agent toward a direction -- is invisible to the process and
    has to be declared. An honest count with reasons a judge can read is worth
    more than an unsupported claim of zero.
    """
    root = repo_root()
    ilog = InterventionLog(Path(root))
    rec = ilog.record("manual_note", args.reason)
    print(f"recorded: [{rec.kind}] {rec.reason}")
    print(f"total interventions this run: {ilog.count} — {ilog.summary()}")


# ----------------------------------------------------------- dashboard ----
def cmd_dashboard(args: argparse.Namespace) -> None:
    root = repo_root()
    results_path = str(Path(root, "results.tsv"))
    dashboard_path = str(Path(root, "results_dashboard.html"))
    concepts_path = str(Path(root, "concepts.json"))
    dashboard_mod.write_dashboard(root, results_path, dashboard_path, concepts_path)
    print(f"wrote {dashboard_path}")


# ----------------------------------------------------------------- graph --
def cmd_graph(args: argparse.Namespace) -> None:
    from . import graph as graph_mod
    compiled = graph_mod.build_graph()
    mermaid = compiled.get_graph(xray=True).draw_mermaid()
    Path(args.out).write_text(mermaid, encoding="utf-8")
    print(f"wrote {args.out}")
    print(mermaid)


def main() -> None:
    ap = argparse.ArgumentParser(prog="autoresearch_lg")
    sub = ap.add_subparsers(dest="command", required=True)

    p_setup = sub.add_parser("setup", help="create the run branch, log the baseline")
    p_setup.add_argument("--tag", required=True)
    p_setup.add_argument("--data-dir", default="./KuaiRand-Pure/data")
    p_setup.set_defaults(func=cmd_setup)

    p_run = sub.add_parser("run", help="run the full loop to convergence (needs ANTHROPIC_API_KEY)")
    p_run.add_argument("--tag", required=True)
    p_run.add_argument("--data-dir", default="./KuaiRand-Pure/data")
    p_run.add_argument("--model", default=bootstrap.DEFAULT_MODEL,
                        help="propose's LLM — claude-* dispatches to Anthropic, anything else to "
                             "OpenAI (needs OPENAI_API_KEY), e.g. claude-haiku-4-5 for a cheap smoke-test")
    p_run.add_argument("--max-iterations", type=int, default=50,
                        help="absolute cap on total experiments (technical-plan.md default: 50)")
    p_run.add_argument("--max-wall-hours", type=float, default=6.0,
                        help="absolute wall-clock budget from `setup` time (default: 6h)")
    p_run.add_argument("--epsilon", type=float, default=0.002, help="noise floor / plateau threshold")
    p_run.add_argument("--n-plateau", type=int, default=3, help="consecutive non-improving iterations to converge")
    p_run.add_argument("--retry-cap", type=int, default=3, help="blind retries on 'error' before forced pivot")
    p_run.add_argument("--tune-cap", type=int, default=3, help="tune loops on one concept before forced expand")
    p_run.set_defaults(func=cmd_run)

    p_fin = sub.add_parser("finalize",
                           help="write submission.csv + resource_report.json from disk")
    p_fin.add_argument("--data-dir", default="./KuaiRand-Pure/data")
    p_fin.add_argument("--model", default=bootstrap.DEFAULT_MODEL)
    p_fin.add_argument("--max-iterations", type=int, default=50)
    p_fin.add_argument("--max-wall-hours", type=float, default=6.0)
    p_fin.add_argument("--epsilon", type=float, default=0.002)
    p_fin.add_argument("--n-plateau", type=int, default=3)
    p_fin.add_argument("--retry-cap", type=int, default=3)
    p_fin.add_argument("--tune-cap", type=int, default=3)
    p_fin.add_argument("--reason", default="manual finalize")
    p_fin.add_argument("--force", action="store_true",
                       help="rebuild even if resource_report.json already exists")
    p_fin.set_defaults(func=cmd_finalize)

    p_note = sub.add_parser("note", help="record a manual intervention (Deliverable 3)")
    p_note.add_argument("reason", help="what you did and why, in one line")
    p_note.set_defaults(func=cmd_note)

    p_dash = sub.add_parser("dashboard", help="regenerate results_dashboard.html")
    p_dash.set_defaults(func=cmd_dashboard)

    p_graph = sub.add_parser("graph", help="print/save the graph structure as mermaid (incl. sub-graph internals)")
    p_graph.add_argument("--out", default="autoresearch_lg_graph.mmd")
    p_graph.set_defaults(func=cmd_graph)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
