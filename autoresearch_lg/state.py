"""Shared state schema for the autoresearch agent.

Matches technical-plan.md's ResearchState: one object threads through the
main loop and all three sub-graphs (propose/experiment/critic). Extended
with the practical infra fields (paths, config) our harness needs that the
plan left as implementation detail.
"""
from __future__ import annotations

from typing import Literal, TypedDict

Mode = Literal["tune", "expand", "pivot", "repair"]
Outcome = Literal["improved", "parity", "failed", "error"]
# "parity" (|delta| <= epsilon) is distinct from "failed" (a clear regression):
# not-better is not the same as worse, and a first implementation of a good
# concept usually lands at parity because it is untuned. See
# critic.classify_outcome for the measured case that motivated it.
# "maxed" from the plan's 4-outcome table is not a separate classify_outcome
# label here — it's router's escalation of "improved" once tune_count hits
# tune_cap (see router() in graph.py). The mode ("expand") carries that
# distinction in logs; classify_outcome only needs the 3-way split.


class Experiment(TypedDict):
    iteration: int
    concept_id: str
    concept: str
    hypothesis: str
    description: str
    files_changed: list[str]
    checkpoint_id: int        # sqlite index row id
    exp_dir: str               # runs/<name> — where this experiment's actual files live
    seed: int
    metrics: dict            # {valid_primary, test_primary}
    outcome: Outcome
    mode: Mode
    error: str | None
    tokens_in: int
    tokens_out: int
    wall_clock_s: float
    elapsed_total_s: float


class ConceptAttempt(TypedDict):
    checkpoint_id: int | None
    valid_primary: float
    outcome: Outcome


class Concept(TypedDict):
    id: str
    statement: str
    rationale: str
    status: str               # 'active' | 'closed'
    closed_reason: str        # '' | 'expanded (maxed out after N tunes)' | 'pivoted (...)'
    opened_at_iteration: int
    attempts: list[ConceptAttempt]


class ResearchState(TypedDict):
    # ---- fixed infra, set once at setup ----
    repo_root: str
    data_dir: str
    editable_files: list[str]
    results_path: str
    log_path: str             # runs.jsonl — the required AIDE-style log
    dashboard_path: str
    concepts_path: str        # concepts.json
    store_path: str           # sqlite index over runs/ folders (see tools.py)
    system_prompt: str        # static rules — kept short; README is NOT embedded (see bootstrap.py)
    model: str                # propose's LLM, e.g. "gpt-5.5" or "claude-opus-5" (claude-* -> Anthropic, else OpenAI)

    # ---- run-level config (CLI-overridable, defaults match baseline_scores.json) ----
    max_iterations: int
    max_wall_seconds: float
    epsilon: float
    n_plateau: int
    retry_cap: int
    tune_cap: int

    # ---- carried across iterations ----
    history: list[Experiment]
    concepts: list[Concept]
    active_concept_id: str    # "" when none — propose must open a new one
    baseline_valid_primary: float  # the FM baseline from baseline_scores.json — fixed, what we're trying to beat
    baseline_test_primary: float   # never changes during a run, unlike best_* below
    best_valid_primary: float      # current best achieved so far; starts equal to baseline_valid_primary
    best_test_primary: float
    best_checkpoint_id: int
    best_exp_dir: str         # "" until something beats the baseline; else runs/<name>
    iteration: int
    retry_count: int          # consecutive 'error' outcomes on the current concept
    tune_count: int           # consecutive 'improved'-and-kept-tuning on the current concept
    no_improve_count: int     # for the epsilon/N=3 global plateau convergence rule
    start_time: float
    converged: bool
    mode: Mode
    eda_summary: dict

    # ---- propose sub-graph scratch ----
    context_summary: str
    retrieved_options: str
    idea_concept: str
    idea_hypothesis: str
    idea_description: str
    edited_files: dict
    propose_attempt: int
    diff_valid: bool
    diff_error: str
    tokens_in: int
    tokens_out: int

    # ---- experiment sub-graph scratch ----
    exp_dir: str
    seed_used: int
    run_stdout: str
    step_failed: bool
    failure_step: str
    failure_error: str
    wall_seconds: float
    valid_primary: float
    test_primary: float
    valid_metrics: dict       # {GAUC, nDCG@5, primary} — Deliverable 4 is per-metric
    test_metrics: dict
    diff: str                 # unified diff applied this iteration (Deliverable 3)

    # ---- critic sub-graph scratch ----
    delta: float
    outcome: Outcome

    # ---- router scratch ----
    retry_now: bool
    repair_error: str         # traceback handed to propose in mode='repair'
    repair_exp_dir: str       # the folder holding the code that actually failed
    llm_unavailable: str      # non-empty => propose is terminally down; ship what we have
    stop_reason: str          # why the loop ended, for the resource report

    # ---- finalize output ----
    resource_report: dict
