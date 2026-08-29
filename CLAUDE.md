# CLAUDE.md

Guidance for Claude Code when working in this repo.

## What this is

**Tippytop** — TikTok TechJam 2026 Track 2. An **autonomous ML research agent**
for **within-user ranking** on the **KuaiRand-Pure** short-video dataset.

The task reduces to: produce a `scores` array (one float per eval row) → hand it
to the frozen `evaluate()` → get `GAUC`, `nDCG@5`, `primary` (= mean of the two).
Only *within-user relative order* matters.

**Goal:** beat the FM baseline **primary 0.5946** (test). Oracle ceiling is
**0.8645**, not 1.0 — measure headroom against 0.8645 (~0.27 remaining).

This is a **monorepo with two lanes** sharing one frozen scoring script. Read
[`ARCHITECTURE.md`](ARCHITECTURE.md) before making structural changes.

## Golden rules

1. **Never edit `evaluate.py`.** It is the scoring spec, and there is exactly one
   copy, at the repo root. `data.py` / `baseline.py` / `submit.py` sit beside it;
   the agent edits *copies* of those in `runs/`, never the originals. From
   `tippytop`, reach them only through `src/tippytop/kit.py`.
2. **Iterate on `valid`; use `test` only for final picks.** A change is real only
   if valid `primary` improves by **> 0.002** (FM's seed std is 0.0008).
3. **Compare like with like.** A ranking loss requires grouped batching, which
   costs ≈ **−0.0023** on its own. Measure it against a pointwise control at the
   *same* batching, never against the raw FM baseline — otherwise a correct
   implementation reads as a failure. (Measured; `results/leaderboard.md`.)
4. **Sanity first:** `--model random` must score test primary ≈ 0.4754. If not,
   the harness is broken — fix that before trusting anything else.
5. Log measured runs in `results/leaderboard.md`.
6. Own branch `dev/<name>`; PR into the integration branch.

## Layout

```
evaluate.py            FROZEN scoring spec — never edit. ONE copy, at root.
data.py baseline.py    the kit. agent edits copies under runs/, not these
submit.py              submission builder + checker
baseline_scores.json   published scores, seed variance, convergence constants

autoresearch_lg/       THE AGENT (LangGraph) — primary deliverable
  graph.py             main loop + router (tune / expand / pivot)
  propose.py           think: prompt assembly, LLM call, source validation
  context.py           build_context (history) + HEADROOM (candidate directions)
  experiment.py        do: apply, train, score — every step has a failure branch
  critic.py            judge: compare, keep/revert, classify, write run log
  bootstrap.py         CONFIG_DEFAULTS + CONSTRAINTS (the system prompt)
  state.py             ResearchState threaded through every node
  tools.py             subprocess / git / results.tsv / checkpoints
  cli.py dashboard.py

src/tippytop/          THE MODEL LIBRARY — fast manual experiments
  kit.py               the only import path to the root kit
  config.py            paths + baseline/oracle numbers + convergence rule
  cli.py               run / submit / check / score
  data/dataset.py      load+encode once into Dataset(splits, enc, dim)
  losses/ranking.py    pointwise, listwise, bpr, hybrid + group_bounds
  models/base.py       Model contract: fit(data) / predict(data, split) -> scores
  models/__init__.py   registry — import new model modules here
  models/fm.py         FM baseline, ported from baseline.py
  models/fm_rank.py    fm_listwise / fm_bpr / fm_hybrid
  models/ensemble.py   fm_seedavg / fm_blend / fm_diverse (rank-averaged)
  training/runner.py   shared train/evaluate loop + leaderboard logging
  submission.py        write_submission / read_submission
  agent/               EARLIER agent (Gemini, linear loop) — see ARCHITECTURE.md

tests/                 34 tests, both lanes
results/leaderboard.md shared scoreboard
docs/                  tutorial.md, project-structure.md, kit/ (organizers' docs)
KuaiRand-Pure/         the data (gitignored)     runs/  agent output (gitignored)
```

## Commands

```bash
uv sync                                          # or uv pip install -e ".[dev]"
bash scripts/download_data.sh                    # or scripts/download_data.ps1

# the agent
uv run python -m autoresearch_lg.cli graph       # free, prints the loop
uv run python -m autoresearch_lg.cli setup --tag <fresh>
uv run python -m autoresearch_lg.cli run   --tag <fresh> --max-iterations 2
uv run langgraph dev --no-browser                # Studio; submit {} as input

# models by hand
uv run python -m tippytop run    --model fm --no-log
uv run python -m tippytop submit --model fm --split test --out results/submissions/fm.csv
uv run python -m tippytop check  <csv> --split test
uv run pytest tests/ -q                          # 34 tests
```

`--model` = `fm` | `pop` | `random` | `fm_listwise` | `fm_bpr` | `fm_hybrid` |
`fm_seedavg` | `fm_blend` | `fm_diverse` | any registered name.
Flags: `--k --lr --epochs --seed --alpha --list_size --groups_per_batch`.

## Adding a model

1. New file `src/tippytop/models/<name>.py`, subclass `Model`, decorate
   `@register("<name>")`, implement `fit(data)` and
   `predict(data, split) -> np.ndarray` (one score per row, in row order).
2. Add the module to the import line at the bottom of `models/__init__.py`.
3. `uv run python -m tippytop run --model <name>`. No other file changes.

Model-agnostic losses go in `losses/`, not inside a model.

## Where the score is / isn't

**Measured dead ends** — don't retry: bigger embedding dim; pure user-side
features (constant within a user → provably zero effect on within-user order).

**Narrower than it looks:** the organizers' "static features yield nothing"
ablation (`ablation_features.py`) only reads `video_features_basic_pure.csv` and
only tests four categorical IDs. `video_features_statistic_pure.csv` — 30+
*continuous* per-video engagement columns — was never tested. Prefer computing
equivalent aggregates from the **train split only** (leak-free) over using the
shipped file, whose aggregation window is unknown.

**Open headroom**, roughly ranked: recovering the −0.0023 batching penalty (the
ranking-loss gain is already proven), train-only item aggregates, within-session
position from `time_ms`, multi-task on `is_click`/`is_like`/`play_time_ms`,
censored watch-time (D2Q-style duration-bucket quantiles), LambdaRank via
LightGBM. See `task.md` and `results/leaderboard.md`.

## Environment notes

- Python 3.11+, managed with **uv**. Baseline is numpy-only and CPU; FM ~60s.
  `[models]` extra adds LightGBM + scikit-learn — installed but the agent's
  `CONSTRAINTS` prompt still says numpy-only, so update both if you want it used.
- Windows dev box; PowerShell and bash both available. Prefer forward-slash paths.
- `KuaiRand-Pure/`, `runs/`, `*.egg-info/`, `results/submissions/*.csv`, and
  agent artifacts (`results.tsv`, `concepts.json`, `checkpoints.db`) are
  gitignored — don't commit them.
- Verified reproductions: random test primary 0.4754; FM valid 0.6015 / test
  0.5953 (seed 0). Over 10 seeds: valid 0.6015 ± 0.0006, test 0.5949 ± 0.0008 —
  seed 42 is a full sigma high, so don't quote it as "the baseline".
- `autoresearch_lg` needs `ANTHROPIC_API_KEY` (default model `claude-sonnet-5`).
  `src/tippytop/agent` needs `GEMINI_API_KEY`, except in `--llm mock`.
