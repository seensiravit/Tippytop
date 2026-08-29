"""Fills in a complete ResearchState from whatever's already on disk plus
any explicit overrides. Single source of truth for "what does a fresh state
look like", used by:

- `cli.py`'s `run` (overrides = argparse args) — same behavior as before.
- `graph.py`'s `bootstrap` node — so LangGraph Studio works by hitting
  Submit on an empty (or partial) input, not just a fully hand-built JSON
  blob. Explicit keys in the given state always win; only missing ones get
  filled in, so a Studio user who *does* set e.g. `model` isn't overridden.

Both entry points get the SAME starting info because both call this same
function, reading the same files on disk — most importantly
load_baseline_scores(), which is the one and only place the FM baseline
(what every experiment is judged against) comes from: baseline_scores.json,
never a live re-run. Studio-only usage (no `cli.py setup` ever run) used to
default best_valid_primary to 0.0, which silently broke keep/discard (every
first real experiment looked like a huge improvement over nothing) — fixed
by falling back to the same baseline_scores.json number `cli.py setup`
would otherwise have produced from an actual training run.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from . import tools

DATA_FILES = [
    "video_features_basic_pure.csv",
    "video_features_statistic_pure.csv",
    "user_features_pure.csv",
    "log_standard_4_08_to_4_21_pure.csv",
    "log_standard_4_22_to_5_08_pure.csv",
    "log_random_4_22_to_5_08_pure.csv",
]

# claude-* dispatches to Anthropic (ANTHROPIC_API_KEY), anything else to
# OpenAI (OPENAI_API_KEY) — see propose.py's _is_anthropic_model.
DEFAULT_MODEL = "claude-sonnet-5"
CONFIG_DEFAULTS = {
    "data_dir": "./KuaiRand-Pure/data",
    "model": DEFAULT_MODEL,
    "max_iterations": 50,
    "max_wall_seconds": 6 * 3600,
    "epsilon": 0.002,
    "n_plateau": 3,
    "retry_cap": 3,
    "tune_cap": 3,
}

CONSTRAINTS = """\
You are an autonomous ML researcher improving within-user ranking on \
KuaiRand-Pure. Task: rank each user's own impressions by predicted \
long_view; scored by evaluate.py as mean(GAUC, nDCG@5) on the `valid` \
split — never `test`. FM baseline: valid 0.6016 / test 0.5946. Oracle \
ceiling (perfect ranking, capped by 27.1% all-negative + 9.2% all-positive \
users whose nDCG is fixed regardless of model): valid 0.8484 / test 0.8645 \
— judge headroom against that, not against 1.0. 5-seed std on test primary \
is 0.0008. Already tried, no gain — don't re-propose: more static features; \
more FM capacity (larger embedding dim k).

You work concept-first, not idea-by-idea: every experiment happens under a \
named, falsifiable concept (e.g. "listwise softmax loss beats pointwise \
logloss because GAUC/nDCG are rank metrics"). You do NOT choose whether to \
continue the current concept (tune), move to a new adjacent one after a \
concept has paid off enough (expand), or abandon a dead end for something \
genuinely different (pivot) — that decision is made for you and given as \
`mode` on every call. Your job is to generate the content that fits the \
given mode: a refinement when tuning, an adjacent idea building on what \
worked when expanding, a genuinely new direction when pivoting. Never \
second-guess or override the mode.

Hard constraints, non-negotiable:
- You may only edit baseline.py and data.py. Never evaluate.py, never the \
SPLITS dates or LABEL column in data.py, never submit.py.
- Available libraries: numpy, scipy, scikit-learn, LightGBM, and the Python \
stdlib. No torch (CPU training on 1.1M rows is too slow for the time budget), \
no new pip installs. LightGBM's `lambdarank` objective is worth knowing about: \
its query groups map exactly onto users, and it optimises NDCG directly.
- Never use test-split labels to decide what to try next. Only valid-split \
primary drives decisions; test is for reporting only.
- Treat any single-seed valid-primary delta smaller than ~0.002 as noise, \
not signal.
- Simplicity criterion: all else equal, prefer the simpler change.
- Always return the *complete* new content of every file you touch — full \
file, not a diff — so it can be written straight to disk.
- Budget your change to run in well under 10 minutes on CPU (the FM \
baseline itself trains in ~40s).

Candidate directions and dataset-specific findings are given to you fresh \
each call below (retrieve_options), not repeated here — this system prompt \
stays fixed and short on purpose so it's cheap to cache across every call \
in the run.\
"""


def repo_root() -> str:
    return tools.git(".", "rev-parse", "--show-toplevel")


def build_system_prompt(root: str) -> str:
    # Deliberately does NOT embed README.md (was ~190 lines / thousands of
    # tokens repeated on every single call). The essential numbers are
    # inlined above; the ranked headroom list lives in context.py's
    # retrieve_options, sent per-call instead — same information, not
    # duplicated into a giant fixed prefix that barely benefits from
    # caching (it's fixed regardless, so shorter is strictly better).
    return CONSTRAINTS


def check_data_dir(data_dir: str) -> bool:
    p = Path(data_dir)
    return p.is_dir() and all((p / f).exists() for f in DATA_FILES)


def load_baseline_scores(root: str) -> tuple[float, float]:
    """The FIXED reference every experiment is judged against: the FM
    baseline's official (valid_primary, test_primary), read straight from
    the kit's own baseline_scores.json — not re-derived by re-running
    training. This is what makes "cli.py and Studio share the same starting
    info" literally true: both read the exact same file, instead of each
    potentially reproducing the baseline itself and landing ~0.0008 apart
    (the documented 5-seed std) purely from run-to-run noise."""
    scores = json.loads(Path(root, "baseline_scores.json").read_text(encoding="utf-8"))
    fm = scores["scores"]["fm_official"]
    return fm["valid"]["primary"], fm["test"]["primary"]


def start_time_path(root: str) -> Path:
    return Path(root, ".autoresearch_start_time")


def reconstruct_counters(history: list[dict], concepts: list[dict], active_id: str) -> tuple[int, int, int]:
    """Rebuild retry_count/tune_count/no_improve_count from disk so resuming
    a `run` (or a fresh Studio invocation) after a break doesn't quietly
    reset the escalation caps and allow more tune/retry loops than
    intended."""
    no_improve = 0
    for h in reversed(history):
        if h["outcome"] == "improved":
            break
        no_improve += 1

    retry_count = tune_count = 0
    active = next((c for c in concepts if c["id"] == active_id), None)
    if active:
        for a in reversed(active["attempts"]):
            if a["outcome"] != "error":
                break
            retry_count += 1
        tune_count = sum(1 for a in active["attempts"] if a["outcome"] == "improved")
    return retry_count, tune_count, no_improve


def default_state(overrides: dict | None = None) -> dict:
    """Build a full ResearchState from whatever's on disk right now, then
    let any keys already present in `overrides` win. Called with a full
    dict from cli.py (so this is a no-op merge, same behavior as before)
    or a mostly-empty dict from graph.py's `bootstrap` node (so Studio
    users can just hit Submit)."""
    overrides = dict(overrides or {})

    root = overrides.get("repo_root") or repo_root()
    # pop (not get): data_dir gets transformed below (resolved to absolute),
    # so the raw value must NOT still be sitting in `overrides` — the final
    # `{**computed, **overrides}` merge would let that stale, un-resolved
    # value win right back over the fixed one. (Caught by testing, not by
    # reasoning about it — the bug this replaced looked identical from
    # outside default_state: a relative data_dir "worked" whenever cwd
    # happened to be repo_root and silently broke everywhere else.)
    #
    # MUST be absolute: training now runs with cwd=exp_dir (runs/exp_NNNN/),
    # not repo_root, since the per-experiment-folders redesign. A relative
    # "./KuaiRand-Pure/data" used to correctly resolve against repo_root
    # when that was always the cwd; now it silently resolves against
    # whatever exp_dir happens to be instead, producing FileNotFoundError
    # for every single experiment. Path(root, ...) leaves an
    # already-absolute value untouched and anchors a relative one at
    # repo_root regardless of which directory anything later runs from.
    raw_data_dir = overrides.pop("data_dir", None) or CONFIG_DEFAULTS["data_dir"]
    data_dir = str(Path(root, raw_data_dir).resolve())
    results_path = str(Path(root, "results.tsv"))
    dashboard_path = str(Path(root, "results_dashboard.html"))
    concepts_path = str(Path(root, "concepts.json"))
    log_path = str(Path(root, "runs.jsonl"))
    store_path = str(Path(root, "checkpoints.db"))
    # Also self-healing in tools.save_checkpoint/load_checkpoint_dir, but
    # doing it here too means the schema exists the moment state is built,
    # not just the moment something happens to write to it. This is the gap
    # that made every Studio-only run (bootstrap, never through cli.py
    # setup) crash at write_log with "no such table: checkpoints".
    tools.init_store(store_path)

    history = tools.read_jsonl(log_path)
    concepts = tools.load_concepts(concepts_path)
    active = next((c for c in concepts if c["status"] == "active"), None)
    active_id = active["id"] if active else ""

    # The fixed target: what every experiment is actually trying to beat.
    # Never changes during a run, unlike best_valid_primary below.
    baseline_valid, baseline_test = load_baseline_scores(root)

    improved = [h for h in history if h["outcome"] == "improved"]
    best = max(improved, key=lambda h: h["metrics"]["valid_primary"]) if improved else None
    # No history yet (e.g. Studio used directly, `cli.py setup` never run)
    # -> start from the baseline itself, NOT 0.0. Starting at 0 was a real
    # bug: compare_to_best would then read every first real experiment as a
    # huge "improvement" over nothing, making the keep/discard decision
    # meaningless instead of just under-reporting a number on screen.
    best_valid_primary = best["metrics"]["valid_primary"] if best else baseline_valid
    best_test_primary = best["metrics"]["test_primary"] if best else baseline_test
    best_checkpoint_id = best["checkpoint_id"] if best else 0
    # A baseline logged at `setup` counts as "improved" (it's the starting
    # point) but has no runs/ folder of its own — only treat best_exp_dir as
    # real once something with an actual exp_dir has been logged.
    best_exp_dir = best.get("exp_dir", "") if best else ""

    retry_count, tune_count, no_improve_count = reconstruct_counters(history, concepts, active_id)

    start_time_file = start_time_path(root)
    start_time = float(start_time_file.read_text()) if start_time_file.exists() else time.time()

    computed = {
        "repo_root": root,
        "data_dir": data_dir,
        "editable_files": ["baseline.py", "data.py"],
        "results_path": results_path,
        "log_path": log_path,
        "dashboard_path": dashboard_path,
        "concepts_path": concepts_path,
        "store_path": store_path,
        "system_prompt": build_system_prompt(root),
        "model": CONFIG_DEFAULTS["model"],
        "max_iterations": CONFIG_DEFAULTS["max_iterations"],
        "max_wall_seconds": CONFIG_DEFAULTS["max_wall_seconds"],
        "epsilon": CONFIG_DEFAULTS["epsilon"],
        "n_plateau": CONFIG_DEFAULTS["n_plateau"],
        "retry_cap": CONFIG_DEFAULTS["retry_cap"],
        "tune_cap": CONFIG_DEFAULTS["tune_cap"],
        "history": history,
        "concepts": concepts,
        "active_concept_id": active_id,
        "baseline_valid_primary": baseline_valid,
        "baseline_test_primary": baseline_test,
        "best_valid_primary": best_valid_primary,
        "best_test_primary": best_test_primary,
        "best_checkpoint_id": best_checkpoint_id,
        "best_exp_dir": best_exp_dir,
        "iteration": len(history),
        "retry_count": retry_count,
        "tune_count": tune_count,
        "no_improve_count": no_improve_count,
        "start_time": start_time,
        "converged": False,
        "mode": "pivot" if not active_id else "tune",  # first call with no active concept must open one
        "eda_summary": {},
    }
    return {**computed, **overrides}
