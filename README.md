# Tippytop

**TikTok TechJam 2026 — Track 2: Autonomous ML Research Agent**

An LLM agent that runs the ML iteration loop on its own — reads the problem,
writes experiment code, trains, scores, reflects, and decides what to try next —
on **KuaiRand-Pure** within-user ranking.

Goal: beat the official Factorization Machine baseline of **primary 0.5946**
(test). Judge progress against the oracle ceiling of **0.8645**, not 1.0 — 36.3%
of users cannot be ranked by any model.

New here? Read [`ARCHITECTURE.md`](ARCHITECTURE.md), then
[`docs/tutorial.md`](docs/tutorial.md).

---

## Quick start

```bash
uv sync                          # or: uv pip install -e ".[dev]"
cp .env.example .env             # then add ANTHROPIC_API_KEY
bash scripts/download_data.sh    # or: powershell scripts/download_data.ps1
uv run pytest tests/ -q          # 200 tests
```

Run a model by hand:

```bash
uv run python -m tippytop run --model fm --no-log
uv run python -m tippytop run --model fm_ffm --no-log
uv run python -m tippytop submit --model fm --split test --out results/submissions/fm.csv
```

---

## Running the agent

No venv activation — `uv run` handles it. Two ways in.

### A. Command line

```bash
uv run python -m autoresearch_lg.cli graph                 # free: prints the loop, no API calls
uv run python -m autoresearch_lg.cli setup --tag run1      # ~65s, reproduces the FM baseline
uv run python -m autoresearch_lg.cli run   --tag run1 --max-iterations 2
uv run python -m autoresearch_lg.cli dashboard             # read the results back
```

`setup` creates a git branch `autoresearch/<tag>` and switches to it, and refuses
to reuse a tag — **pick a fresh tag per run**. It is also the gate: if the
baseline does not reproduce, stop and fix that before trusting any later number.

Drop `--max-iterations` for the full 50-iteration / 6-hour run. Safe to Ctrl+C at
any point — every completed iteration is already durable on disk.

### B. LangGraph Studio (visual)

```bash
uv run langgraph dev --no-browser
```

It prints an API URL and a Studio UI URL; ctrl-click the Studio one. Then:

1. In the **Input** panel, click **View Raw**
2. Replace the contents with `{"max_iterations": 2}`
3. **Submit**

The form shows every state field as "Required", but you do not fill them in —
the `bootstrap` node exists precisely so a near-empty input works, filling in
repo root, data path and config defaults.

> ⚠️ **Always pass `max_iterations` in Studio.** A bare `{}` inherits the
> defaults — **50 iterations or 6 hours** — and Studio has no equivalent of the
> CLI's `--max-iterations` flag, so there is nothing to remind you. It will keep
> calling the LLM unattended. Roughly $0.08 per iteration on `claude-sonnet-5`,
> so a forgotten `{}` is ~$4 and several hours.

Other fields override the same way, e.g.
`{"max_iterations": 5, "model": "claude-opus-5"}`.

To stop a run: use the stop control on the thread, or Ctrl+C the
`langgraph dev` terminal. Completed iterations are already durable on disk.

Nodes light up as they run and you can click any of them to inspect the state
going in and out, which is the reason to prefer Studio while learning the loop.
Use the CLI for long runs.

> **Cost.** `graph`, `dashboard` and `setup` make no LLM calls. Everything from
> `propose` onward does: one Claude call plus a training run per iteration.
> Start with 2 iterations and confirm `runs.jsonl` looks right before committing
> to 50.

> The "LangSmith API key missing" banner in Studio is harmless — that is optional
> cloud tracing, not required for any part of the run.

### What to check after a run

```bash
uv run python -m autoresearch_lg.cli dashboard
```

`runs.jsonl` must contain, per iteration: hypothesis, code diff, resulting
metrics, and any error/recovery event — these are graded deliverables, so verify
the shape on a 2-iteration run rather than after a 6-hour one.

---

## The task

| | |
|---|---|
| Task | **Within-user ranking** — each user ranks only their own logged impressions |
| Label | `long_view` (native 0/1 column) |
| Metrics | `GAUC` and `nDCG@5`; **primary = mean of the two** |
| Splits | train `20220408–20220421` / valid `20220422–20220428` / test `20220429–20220508` |
| Budget | 50 iterations or 6 h wall-clock; converged at ε = 0.002 over N = 3 |

Only the **relative order of scores within a user** matters — never absolute
values, never comparisons across users.

| Model | GAUC | nDCG@5 | primary (test) |
|---|---|---|---|
| random (sanity floor) | 0.4996 | 0.4511 | 0.4753 |
| item popularity | 0.6308 | 0.5121 | 0.5715 |
| **FM — the row to beat** | **0.6610** | **0.5282** | **0.5946** |
| oracle (perfect ranking) | 1.0000 | 0.7289 | **0.8645** |

---

## Layout

```
evaluate.py  data.py  baseline.py  submit.py    the frozen kit — ONE copy
autoresearch_lg/                                the agent (LangGraph)
src/tippytop/                                   the model library
tests/  results/  docs/  scripts/  experiments/
KuaiRand-Pure/                                  the data      (gitignored)
runs/                                           agent output  (gitignored)
```

Two lanes, one scoring script. Full breakdown in [`ARCHITECTURE.md`](ARCHITECTURE.md).

---

## Rules

- **Never edit `evaluate.py`** — it is the scoring spec. Import via `tippytop.kit`.
- **Iterate on `valid`.** `test` is for reporting the final pick only.
- **Δvalid > 0.002 or it did not happen** — FM's seed std is 0.0008.
- **Compare like with like.** A ranking loss needs grouped batching, which costs
  ≈ −0.0023 by itself; measure it against a pointwise control at the *same*
  batching, never against the raw FM baseline.
- **Log every run** in [`results/leaderboard.md`](results/leaderboard.md).

---

## Current state

Nothing has beaten the baseline decisively yet.

Rank-averaged ensembles give a replicated **+0.0013** — real (verified across two
disjoint seed groups, with a monotonic dose-response curve) but below the 0.002
threshold, so it is variance reduction rather than a modelling win.

Against a matched control, the ranking loss is worth **+0.0010 to +0.0017** in 4
of 4 configurations. It loses in absolute terms only because the batching it
requires costs more than it gains — recovering that is open work. Full numbers in
[`results/leaderboard.md`](results/leaderboard.md).

---

## Before submitting

Done:

- [x] **One agent.** `src/tippytop/agent/` (the earlier Gemini lane) is removed;
      `autoresearch_lg/` is the deliverable. `interventions.py` and `redact.py`
      were moved to `src/tippytop/runlog/` first and wired into the surviving
      agent — see [`ARCHITECTURE.md`](ARCHITECTURE.md) → *Decisions*.
- [x] **The intervention count is measured, not asserted.** Resumes are detected
      automatically; anything else is recorded with
      `python -m autoresearch_lg.cli note "<reason>"`. `finalize` writes the
      count and every reason into `resource_report.json`.
- [x] **The run survives things it does not control.** Transient provider
      errors retry with backoff; a dead provider routes to `finalize` instead of
      raising; a crashing experiment is repaired with its own traceback rather
      than rerolled at a new seed; and `run` finalizes in a `finally`, so the
      graded artifacts get written on every exit path. `ARCHITECTURE.md` →
      *Failure policy*. If a run ever does end without them:
      `python -m autoresearch_lg.cli finalize`.
- [x] **No test metric can reach the proposing model.** The crashed-run stdout
      tail — the only path that carried one — is scrubbed before it becomes
      `failure_error`. Covered by `tests/test_run_integrity.py`.

Still to do, and each needs a completed agent run:

- [ ] Run the agent to convergence on the real data:
      `python -m autoresearch_lg.cli setup --tag final` then `run --tag final`
- [ ] `python scripts/package_final_run.py` — gates the artifacts, then copies
      `runs.jsonl` / `resource_report.json` / `submission.csv` / `results.tsv` /
      `concepts.json` / `interventions.jsonl` into `results/final_run/`.
      It refuses on a missing artifact, a rejected CSV, a run short enough to be
      a smoke test, or an intervention count that disagrees with its own log.
- [ ] `git add results/final_run && git commit` — a grader cannot see files that
      are not in the repository
- [ ] Devpost description — draft in
      [`docs/devpost.md`](docs/devpost.md); fill the bracketed run numbers from
      `resource_report.json`

## Reproducing our result

```bash
python -m pip install -e .            # or: uv sync
python scripts/download_data.py       # KuaiRand-Pure, ~46 MB, from Zenodo
python -m pytest tests/ -q            # 200 tests

# 1. sanity — if this is not ~0.4754, nothing else is trustworthy
python -m tippytop run --model random --no-log

# 2. reproduce the official baseline (test primary 0.5946 +- 0.0008)
python -m tippytop run --model fm --no-log

# 3. our best single model, and the ensemble
python -m tippytop run --model ffm --no-log
python -m tippytop run --model fm_blend --no-log

# 4. the agent, end to end (needs ANTHROPIC_API_KEY in .env)
python -m autoresearch_lg.cli setup --tag final
python -m autoresearch_lg.cli run   --tag final

# 5. collect the graded artifacts (refuses if anything is not gradeable)
python scripts/package_final_run.py
```

Committed artifacts from our submission run are in
[`results/final_run/`](results/final_run/): `runs.jsonl` (per-iteration
hypothesis, **code diff**, metrics, errors and recovery), `resource_report.json`
(tokens, wall-clock, iterations, intervention count, results table),
`submission.csv`, `results.tsv`, `concepts.json`, `interventions.jsonl`,
`recovery.jsonl`. Every measured comparison is in
[`results/leaderboard.md`](results/leaderboard.md).

## Limitations, and what we would do with more time

**We are noise-limited, not idea-limited.** Seed-to-seed variation is now
0.0002–0.0003, but evaluation noise — resampling the ~24k evaluation users — is
≈0.0008. More seeds cannot tighten that. Any future gain below ~0.002 is not
distinguishable from noise at this sample size, which is why we report
paired-bootstrap intervals rather than differences of scalars. With more time we
would rank-average across seeds *before* bootstrapping, which is the only lever
that actually narrows the interval.

**Validation→test transfer is unmeasured.** We treat a validation gain as
predictive of a test gain but have never estimated the slope, and the evidence we
have is not 1:1 (FFM: +0.0009 valid, +0.0019 test). `src/tippytop/stats/
transfer.py` can estimate it; we do not yet have enough distinct models to do so
reliably.

**`time_ms` is untouched.** Within-session position varies *within* a user, so
unlike every user-side aggregate it can reorder — the most obvious remaining
direction, and one the agent has not been given.

**The ensemble choice is weaker than the ensemble.** 6FM+6FFM beats 6FFM alone by
+0.0005 on test, inside the noise band, and that comparison used test. "FFM plus
ensembling beats FM" is replicated on validation across disjoint seed halves; the
choice between the two ensembles is not.

**The agent's search is breadth-first over concepts, not depth-first over one.**
`tune_cap=3` bounds refinement before it pivots. On a benchmark where the
remaining headroom is small and the noise floor is high, more depth per concept
would probably beat more concepts — but that is a hypothesis, not a measurement.

**We did not attempt the bonus benchmarks** (KuaiRand-1k, KuaiRand-27k). The
pipeline is dataset-agnostic below `data.py`, but we chose to spend the budget on
evidence quality for the required benchmark rather than coverage.

## Team

<!-- Required deliverable: fill this in before submitting. -->

| Member | Contribution |
|---|---|
| *(name)* | *(e.g. LangGraph agent loop, router and convergence rule)* |
| *(name)* | *(e.g. FFM and ensemble models, leaderboard experiments)* |
| *(name)* | *(e.g. statistical layer: paired bootstrap, power analysis)* |
| *(name)* | *(e.g. failure policy, integrity guards, test suite)* |

Working agreement: own branch `dev/<name>`, then PR into the integration branch.
Put your best valid/test primary in the PR description.
