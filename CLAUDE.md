# CLAUDE.md

Guidance for Claude Code when working in this repo.

## What this is

**Tippytop** — TikTok TechJam 2026 entry. A **within-user ranking** model on the
**KuaiRand-Pure** short-video dataset. Team of 5.

The task reduces to: produce a `scores` array (one float per eval row) → hand it
to the kit's frozen `evaluate()` → get `GAUC`, `nDCG@5`, `primary` (= mean of the
two). Only *within-user relative order* of scores matters.

**Goal:** beat the FM baseline **primary 0.5946** (test). Oracle ceiling is
**0.8645**, not 1.0 — measure headroom against 0.8645 (~0.27 remaining).

## Golden rules

1. **Never edit `kuairand-starter-kit/`.** It is vendored and frozen. `evaluate.py`
   is the scoring spec; `data.py`/`baseline.py`/`submit.py` are the reference. Reach
   them only through `src/tippytop/kit.py`, which re-exports `load`, `encode`,
   `evaluate`, `FIELDS`, `SPLITS`, `LABEL`.
2. **Iterate on the `valid` split; use `test` only for final picks.** A change is
   real only if valid `primary` improves by **> 0.002** (FM's seed std is 0.0008).
3. **Sanity first:** `--model random` must score test primary ≈ 0.4754. If not, the
   harness is broken — fix that before anything else.
4. Log measured runs in `results/leaderboard.md`.
5. Each person works on their own branch `dev/<name>`; PR into `main`.

## Layout

```
kuairand-starter-kit/        VENDORED, FROZEN — do not edit
src/tippytop/
  kit.py                     bridge to the frozen kit (only way to import it)
  config.py                  paths + baseline/oracle numbers + convergence rule
  cli.py / __main__.py       the CLI (run / submit / check / score)
  data/dataset.py            load+encode once into a Dataset(splits, enc, dim)
  data/features.py           feature engineering (STUB)
  data/sequences.py          per-user history for sequence models (STUB)
  losses/ranking.py          pointwise done; BPR / listwise are STUBS (top headroom)
  models/base.py             Model contract: fit(data) / predict(data, split)->scores
  models/__init__.py         registry; import new model modules here so @register runs
  models/fm.py               FM baseline (0.5946) — DONE, ported from baseline.py
  models/popularity.py       pop baseline — DONE
  models/random_model.py     random baseline — DONE
  training/runner.py         shared train/evaluate loop + leaderboard logging
  submission.py              write_submission + read_submission (validation)
  agent/                     autonomous ML research agent (Track 2 core) — DONE machinery
    orchestrator.py          AIDE-style loop: seed -> improve/debug -> score valid -> finalize
    llm/{base,mock,gemini}.py  client ABC; offline mock; stdlib-only Gemini REST
    {prompts,parsing,contract,guard,sandbox,scoring,convergence,journal,cli}.py
scripts/                     download_data.{sh,ps1}, run_experiment.py, make_submission.py
results/leaderboard.md       shared scoreboard
tests/test_harness.py        random ≈ 0.475 sanity check
docs/                        tutorial.md, project-structure.md
task.md                      goal + measured strategy   techjam2026_..._agent.md  official statement
```

## Commands

```bash
pip install -e ".[dev]"                          # enables `python -m tippytop`
bash scripts/download_data.sh                    # or scripts/download_data.ps1

python -m tippytop run    --model fm             # train + evaluate valid & test
python -m tippytop submit --model fm --split test --out results/submissions/fm_test.csv
python -m tippytop check  <csv> --split test     # validate format/alignment
python -m tippytop score  <csv> --split valid    # validate + score (valid only)
python -m tippytop agent  --llm mock --max-iters 3   # autonomous agent, offline (no API)
python -m tippytop agent  --llm gemini --max-iters 50  # needs GEMINI_API_KEY env var
python -m pytest tests/ -v                        # harness + agent sanity (26 tests)
```

Without an install, use the wrappers: `python scripts/run_experiment.py --model fm`
and `python scripts/make_submission.py --model fm --split test --out ...`.
`--model` = `fm` | `pop` | `random` | any registered name. FM flags: `--k --lr --epochs --seed`.

## Adding a model

1. New file `src/tippytop/models/<name>.py`, subclass `Model`, decorate
   `@register("<name>")`, implement `fit(data)` and `predict(data, split) -> np.ndarray`
   (one score per row of `data.splits[split]`, in row order).
2. Add the module to the import line at the bottom of `models/__init__.py`.
3. `python -m tippytop run --model <name>`. No other files change.
Model-agnostic losses go in `losses/`, not inside a model.

## Where the score is / isn't

Measured dead-ends (don't retry): adding static features, bigger embedding dim, and
pure user-side features (constant within a user → zero effect on within-user order).
Real headroom, ranked: **(1) ranking loss** (BPR/listwise — pointwise↔ranking
mismatch, no new data, start here), (2) user behavior sequences, (3) multi-task
(`is_click/like/...`, `play_time_ms`), (4) watch-time (censored regression),
(5) deeper models. See `task.md`.

## Environment notes

- Python 3.14, numpy-only baseline (no torch/pandas/sklearn). Runs on CPU; FM ~50s.
- Windows dev box; both PowerShell and bash available. Prefer forward-slash paths.
- Data (`KuaiRand-Pure/`), `*.egg-info/`, and `results/submissions/*.csv` are
  git-ignored — don't commit them.
- Verified reproductions: random test primary 0.4754; FM test primary ~0.595 (seed 0).
- Agent (`tippytop.agent`): dependency-free (stdlib `urllib` for Gemini REST, no SDK).
  Default model `gemini-3.5-flash-lite` (the API retired `gemini-2.5-flash-lite` for new
  keys). Key via `GEMINI_API_KEY` env var — never commit it. Run artifacts land in
  `results/runs/<run_id>/` (git-ignored). See `docs/agent.md`.
