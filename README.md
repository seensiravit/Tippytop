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
uv run pytest tests/ -q          # 34 tests
```

Run the agent:

```bash
uv run python -m autoresearch_lg.cli graph                # free — prints the loop
uv run python -m autoresearch_lg.cli setup --tag run1     # reproduces the baseline
uv run python -m autoresearch_lg.cli run   --tag run1 --max-iterations 2
uv run langgraph dev --no-browser                         # LangGraph Studio; submit {}
```

Run a model by hand:

```bash
uv run python -m tippytop run --model fm --no-log
uv run python -m tippytop run --model fm_seedavg --no-log
uv run python -m tippytop submit --model fm --split test --out results/submissions/fm.csv
```

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

- [ ] **Remove `src/tippytop/agent/`** — an earlier agent kept only for offline
      `--llm mock` testing during development. The deliverable is *one*
      autonomous agent; shipping two makes a judge guess which produced the
      result. Exact steps in [`ARCHITECTURE.md`](ARCHITECTURE.md) → *Decisions*.
- [ ] Final submission CSV passes `python submit.py --check --split test <csv>`
- [ ] `results/leaderboard.md` reports validation-best GAUC / nDCG@5 and the
      absolute delta over the official baseline
- [ ] Run logs cover, per iteration: hypothesis, code diff, metrics, and any
      error/recovery event
- [ ] Report total LLM tokens, agent wall-clock, and iterations used

## Team

Work on your own branch `dev/<name>`, then PR into the integration branch. Put
your best valid/test primary in the PR description.
