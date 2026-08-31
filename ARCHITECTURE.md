# Architecture

What every folder is, why it exists, and which rules must not be broken.

This repo is a **monorepo with two lanes** that share one frozen scoring script.

```
                        evaluate.py   ← frozen, ONE copy, the task spec
                       ▲            ▲
                       │            │
          autoresearch_lg/        src/tippytop/
          the agent               the model library
          writes its own code     humans test hypotheses
          50 iters, autonomous    registry + tests, minutes
                       │            │
                       └── findings ┘
                        (as prompt text)
```

The agent is the deliverable. The library is where a person proves an idea works
quickly; what it measures then goes into the agent's prompts as *context* — never
as a fixed schedule, which would score as a human-authored curriculum rather than
autonomous research.

---

## The frozen kit — repo root

The organizers' starter kit, kept at the root as **one copy**.

| File | Role |
|---|---|
| `evaluate.py` | **FROZEN. Never edit.** GAUC / nDCG@5 / primary. The entire task spec. |
| `data.py` | Loader, the fixed date splits, feature encoding. The agent edits *copies*. |
| `baseline.py` | FM / popularity / random. The FM row is what we must beat. Agent edits *copies*. |
| `submit.py` | Builds and validates the submission CSV. |
| `baseline_scores.json` | Published scores, seed variance, convergence constants. |
| `ablation_features.py` | The organizers' own feature ablation. |

Both lanes read these same files. `autoresearch_lg` seeds every experiment folder
from them; `tippytop.kit` puts the root on `sys.path` and re-exports `load`,
`encode`, `evaluate`.

> **Why one copy matters.** The two merged branches each shipped their own kit. Two
> `evaluate.py` files in a repo whose task is *defined* by `evaluate.py` is a
> correctness hazard — they drift, and nobody can say which one scored a result.
> The copies were verified logic-identical (only Chinese vs English comments and
> help strings differed); the translated one was kept.

---

## Lane 1 — `autoresearch_lg/` (the agent)

A LangGraph agent that runs the ML iteration loop on its own: propose a change,
run it, score it, decide what to try next, repeat until convergence.

```
bootstrap → eda → propose → experiment → critic → router
                    ↑                                │
                    │        error + retries left ───┘
                    │                                │
                    └─── continue ─ check_convergence ─→ finalize → END
```

| File | Responsibility |
|---|---|
| `graph.py` | The main loop and the **router** — reads the critic's verdict, sets the next mode (tune / expand / pivot), applies escalation caps. |
| `propose.py` | The "think" stage: builds the prompt, calls the LLM, validates the returned source parses. |
| `context.py` | `build_context` (run history) and `retrieve_options` (`HEADROOM` — candidate directions, offered as options, never as a schedule). |
| `experiment.py` | The "do" stage: apply, train, score. Every step has a failure branch. |
| `critic.py` | The "judge" stage: compare to best, keep or revert, classify, write the run log. |
| `bootstrap.py` | `CONFIG_DEFAULTS` (model, iteration caps, ε) and `CONSTRAINTS` (the system prompt). |
| `state.py` | `ResearchState` — the one object threaded through every node. |
| `tools.py` | Subprocess, git, results.tsv, checkpoints. Non-LLM building blocks. |
| `cli.py`, `dashboard.py` | Entry points and the run dashboard. |

**Router modes.** `improved` → *tune* (refine the same concept). `improved` at the
tune cap → *expand* (adjacent concept). `failed` → *pivot* (abandon it).
`error` → retry, then pivot. The router sets the *mode*; the LLM decides *what* to
try. Keeping that split is what makes the run autonomous rather than scripted.

**Isolation.** No git in the loop, and root `baseline.py` / `data.py` are never
written to. Each experiment gets its own folder under `runs/`, so "revert" just
means not reading that folder again.

---

## Lane 2 — `src/tippytop/` (the model library)

Where a human tests a hypothesis in minutes. Every model implements the same
two-method contract, so the runner, submission tooling and leaderboard work
unchanged.

```python
model.fit(dataset)                      # train
scores = model.predict(dataset, split)  # one float per row, in row order
```

| Path | Contents |
|---|---|
| `kit.py` | Bridge to the root kit. The only place the kit is imported from. |
| `config.py` | Paths, baseline constants, convergence parameters. |
| `data/` | `dataset.py` (load + encode once), `features.py`, `sequences.py`. |
| `losses/ranking.py` | `listwise_softmax_grad`, `bpr_grad`, `hybrid_grad`. Model-agnostic — they return the gradient wrt logits. |
| `models/` | One file per model, each `@register("name")`. `fm`, `popularity`, `random_model`, `fm_rank`, `ensemble`. |
| `training/runner.py` | The shared load → fit → score → log loop. |
| `submission.py` | Write and validate submission CSVs. |
| `cli.py` | `python -m tippytop run|submit|check|score` |
| `runlog/` | `interventions.py` — the graded manual-intervention count, derived from a durable JSONL log; `redact.py` — scrubs hidden-test signal out of text bound for a prompt. Both wired into `autoresearch_lg`; see *Decisions*. |

Adding a model is one file plus one import line in `models/__init__.py`.

---

## Supporting directories

| Path | Contents |
|---|---|
| `tests/` | Both suites. `pytest tests/ -q` — 34 tests. |
| `results/` | `leaderboard.md` (every measured run) and `submissions/`. |
| `docs/` | `tutorial.md`, `project-structure.md`, and `kit/` — the organizers' own kit docs. |
| `scripts/` | `download_data.{sh,ps1}`, experiment wrappers. |
| `experiments/configs/` | One YAML per run, for reproducibility. |
| `runs/` | Agent experiment folders. Gitignored. |
| `KuaiRand-Pure/` | The dataset, ~194 MB. Gitignored — run the download script. |

---

## Rules

1. **Never edit `evaluate.py`.** It is the scoring spec. Import it via
   `tippytop.kit` or let the agent's harness call it.
2. **Never tune against `test`.** Iterate on `valid`; `test` is for reporting the
   final pick. The hidden test set is scored once.
3. **Δvalid > 0.002 or it did not happen.** FM's seed std is 0.0008; over many
   iterations, best-of-N on that noise manufactures apparent gains.
4. **Compare like with like.** A ranking loss requires grouped batching, which
   costs ≈ −0.0023 on its own. Measure it against a pointwise control using the
   *same* batching, never against the raw FM baseline — otherwise a correct
   implementation reads as a failure. (Measured; see `results/leaderboard.md`.)
5. **Log every run** in `results/leaderboard.md`.
6. **The agent chooses what to try.** Give it context and findings, not a schedule.

---

## Environment

Managed with `uv`. One `pyproject.toml` declares both packages.

```bash
uv sync                       # or: uv pip install -e ".[dev]"
uv pip install -e ".[models]" # optional: LightGBM + scikit-learn
bash scripts/download_data.sh # or: powershell scripts/download_data.ps1
```

Keys go in `.env` (gitignored) — see `.env.example`.

`[models]` is optional on purpose: the numpy-only path still works with no extra
install, but the organizers permit any open-source library, and LightGBM's
`lambdarank` maps query groups directly onto users.

---

## Decisions

### One agent — the second lane has been removed

`autoresearch_lg/` is **the** agent and the deliverable. An earlier
implementation (linear loop, Gemini) lived at `src/tippytop/agent/` and was kept
during development for offline `--llm mock` smoke tests and as a fallback. It has
been **deleted**: the brief asks for *an* autonomous ML research agent, and a repo
containing two of them — two loops, two providers, two prompt sets — makes a judge
guess which one produced the result, while dragging `GEMINI_API_KEY` and a second
dependency path into the setup instructions for nothing.

Two modules were rescued from it before the delete rather than going down with it,
because neither was ever agent-specific and both are graded:

| module | now at | why it survives |
|---|---|---|
| `interventions.py` | `src/tippytop/runlog/` | Deliverable 3 requires the manual-intervention count; Impact & Relevance (20%) is scored on it |
| `redact.py` | `src/tippytop/runlog/` | scrubs hidden-test signal out of text on its way into a prompt — what makes the walled-validation claim true rather than merely asserted |

Both are now wired into `autoresearch_lg`, which is where they should have been:

- `experiment.run_and_evaluate` scrubs the crashed-run stdout tail before it
  becomes `failure_error`. That string is fed back to the proposing model by
  `context.build_context`, and a run that crashes *after* `baseline.py` prints its
  summary block carries the test primary in exactly those characters. It was the
  one hole in an otherwise validation-only prompt.
- `graph.finalize` reads the durable intervention log and writes
  `manual_interventions`, `intervention_summary` and the full reason list into
  `resource_report.json`.
- `cli run` detects a resume (runs.jsonl already holds iterations) and records it
  whether or not the operator declares it; `cli note "<reason>"` records the ones
  the harness cannot see.
- `tools.RUN_ARTIFACTS` archives `interventions.jsonl` with everything else, so a
  fresh `setup` cannot inherit the previous run's count.

> **A trap worth naming.** The removal recipe this file used to carry said
> `git rm tests/test_agent_*.py`. That glob also matches
> `tests/test_agent_parity_outcome.py`, which tests **`autoresearch_lg`'s own
> router**, not the deleted lane. It has been renamed
> `tests/test_router_parity_outcome.py` so the name no longer sets the trap.

After the removal and the failure-policy work: **200 tests pass** (84 surviving + 16 new in
`tests/test_run_integrity.py`, which covers the two rescued modules at their new
wiring points and the final-run packager's refusals).

### Failure policy — four layers, and the one that used to kill the run

The loop already routed every *expected experiment* failure into a
`FailureRecord`: a bad diff, a training crash, a timeout, an unparseable summary
each had its own edge to `emit_failure`. What it had no policy for were the
failures that are not the experiment's. `autoresearch_lg/resilience.py` holds
that policy; the table is the whole design.

| layer | example | before | now |
|---|---|---|---|
| **L1** transient infra | 429, 529 overloaded, read timeout | one unhandled exception propagated out of two sub-graphs and out of `.stream()` — **the run died** | SDK `max_retries=6` **plus** a LangGraph `RetryPolicy` on `propose` (5 attempts, exponential backoff, jitter) |
| **L2** malformed output | no `tool_use` block, missing key, bad syntax | syntax only; `next(...)` raised `StopIteration` | typed `ProposalError` + payload validation; regenerate up to 3× (AIDE's `search.max_debug_depth`) |
| **L3** defective code | `NameError`, shape mismatch, NaN, timeout | blind reroll at a new seed — a guaranteed no-op on a deterministic bug, at the cost of a full training run each time | classify the error; **reseed only if it looks stochastic**, otherwise **repair** with the traceback fed back (MLE-STAR's debugging module) |
| **L4** terminal | budget spent, provider gone, Ctrl+C, unexpected crash | `finalize` was reachable **only** through clean convergence | `error_handler` routes a dead provider to `finalize`; `cmd_run` finalizes in a `finally`; `cli finalize` rebuilds from disk |

**L4 is the one that mattered.** A run that died at iteration 30 of 50 left
`runs.jsonl` on disk with thirty good iterations and produced no
`submission.csv` and no `resource_report.json` — Deliverables 3 and 4 empty,
roughly 40% of the rubric, for work that had already been done.

Two related bugs found while wiring it:

- **Crashes counted toward the plateau.** `no_improve_count` incremented on any
  non-improvement, so with `n_plateau=3` *three consecutive broken experiments
  converged the whole run* — the agent stopping because it hit bugs, not because
  it was out of ideas. Errors now have their own budget (`retry_cap` → repair →
  pivot) and are skipped by the plateau counter, in `critic.update_counters`
  **and** in `bootstrap.reconstruct_counters`, which must agree or a resumed run
  converges at a different point than an uninterrupted one.
- **The budget check was backward-looking.** It asked whether the budget was
  already spent, so at 5h58m it would start an experiment that runs to the
  10-minute cap, overshoot the stated 6h, and leave nothing for finalize. It now
  asks whether there is room for another experiment *and* for shipping, using a
  rolling **median** experiment time (a mean would let one 600s timeout scare it
  into stopping early) and a 7-minute finalize reserve.

Deliberate non-choices, because the second-order cost is worse than the benefit:

- **No `set_node_defaults(retry_policy=...)`.** Retrying `experiment` would
  silently re-run training on an already-broken idea; retrying `critic` would
  double-write `runs.jsonl`. Only `propose` touches a network the run does not
  control, so only `propose` retries. Asserted in `tests/test_resilience.py`.
- **No node `timeout=`.** LangGraph rejects `timeout` on *sync* nodes at compile
  time, and every node here is sync. The 10-minute experiment cap already lives
  in `tools.RUN_TIMEOUT_SECONDS`, where it belongs.
- **Permanent errors are not retried.** A bad API key will not fix itself, and
  five exponential backoffs before dying spend the wall-clock finalize needs.
  `resilience.is_transient` splits the two by exception class and HTTP status.

Every recovery is appended to `recovery.jsonl` and summarised in
`resource_report.json` (`recovery_events`, `recovery_by_action`, `stop_reason`).
Robustness is 20% of the score and is graded on recovery — a run that recovered
silently is indistinguishable from one that never had a problem.

Sources: MLE-STAR (NeurIPS 2025) for repair-then-fall-back-to-last-known-good;
AIDE `aide/utils/config.yaml` for the debug budget of 3; LangGraph's own
fault-tolerance primitives rather than a hand-rolled loop inside a node, which
the checkpointer cannot see.

### Three seams where a bad proposal looks like a good result

An adversarial read of the loop asks a different question from "does it handle
failures": *where can a plausible-looking proposal produce a result that is
wrong rather than absent?* There are exactly three, because the agent's editable
surface is small — it writes two files into one folder, the harness runs them,
and a regex reads a number back out.

**Seam 1 — the model chooses the paths it writes.** `write_experiment_files` did
`Path(exp_dir, path).write_text(...)` with `path` coming straight from the LLM.
`"../../baseline.py"` resolves outside the folder and overwrites the frozen root
kit; `"evaluate.py"` replaces the scoring spec that `make_experiment_dir` copied
in moments earlier, so the experiment is then scored **by the model's own
evaluator**. That is the highest-reward move available to anything optimising the
number this harness reads, and it would appear in `runs.jsonl` as a triumph.
`tools.safe_experiment_path` now refuses protected names, absolute paths (POSIX
and Windows), and anything that leaves the folder after resolution — and it
validates every path *before* writing any of them, because a half-written
experiment still runs and still gets scored.

**Seam 2 — the harness believes the number the run prints.** `parse_summary` read
`([\d.]+)` with no bound. `primary 99.0` parsed fine and would become the
incumbent permanently: `best_valid_primary` only moves up, every later proposal
is built from that folder, and `finalize` ships it. `mean(GAUC, nDCG@5)` cannot
leave `[0, 1]`, so that is now enforced. The same regex also matched the `1` of
`1e9` and reported a perfect 1.0; a negative lookahead fixes it. `collect_metrics`
reports "no summary block" and "primary outside [0, 1]" as *different* failures,
since the model can only repair what it is told.

**Seam 3 — every later proposal is built from the incumbent's folder.** If
`best_exp_dir` has been deleted, `read_experiment_files` raised a bare `OSError`
inside `llm_generate` — a node whose retry policy would then back off five times
and declare the *provider* dead, which is the wrong diagnosis and the wrong
response to a local, permanent problem. It now falls back to the pristine root
files and **tells the model it is starting from baseline**, rather than silently
handing over baseline code labelled as the current best.

Three smaller ones from the same pass:

- A duplicated path in `files[]` was silently resolved last-wins by the dict
  comprehension. It is now rejected: a training run on code nobody chose costs
  more than one regeneration.
- The stochastic/deterministic classifier is a regex, so
  `ValueError: could not convert string to float: 'nan'` earned a free reseed
  that could not possibly work. Named deterministic exceptions now take
  precedence, and quoted spans are stripped before the stochastic match — a
  quoted token is data, not a numerical event.
- An unsafe path is a defective *proposal*, so it lands on the failure branch
  with its reason attached and becomes a repair, not a traceback.

`tests/test_integrity.py` covers all of these plus the exact epsilon boundaries
(`+0.002` and `-0.002` are both `parity`; only strictly beyond is `improved` /
`failed`).

### Libraries are open — matching the brief, not restricting past it

The problem statement names LightGBM twice as in scope (§2.3 *In scope*; §2.4
*Resource policy*: "use any open-source library … The agent is expected to draw on
whatever published methods it can find"). The only hard rules are **no external
training data** and **no hidden-test access**.

So `CONSTRAINTS` in `bootstrap.py` offers numpy, scipy, scikit-learn and LightGBM,
and `pyproject.toml` carries them behind the optional `[models]` extra. **Torch is
excluded on time budget, not policy** — CPU training over 1.1M rows does not fit
the 10-minute per-experiment cap.

If you change one, change both: the extra controls what is *installed*, the prompt
controls what the agent will *reach for*.
