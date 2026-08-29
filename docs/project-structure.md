# Project Structure — Team Guide

A walkthrough of how this repo is organized and **how we work in it as a team of
5 without stepping on each other**. Read this once before you start.

> TL;DR — the starter kit is frozen and vendored; we build our own `src/tippytop/`
> package around it. Every model boils down to producing a `scores` array that
> gets handed to the kit's `evaluate()`. Work on your own branch `dev/<name>`,
> add files in your area, log results in `results/leaderboard.md`.

---

## 1. The big picture

The whole task reduces to one contract:

```
your model  →  scores (one float per eval row)  →  evaluate()  →  GAUC, nDCG@5, primary
```

`evaluate()` lives in the starter kit and is the **frozen scoring spec** — we
never change it. Because scoring is fully decoupled from the model, five people
can build five completely different approaches and they all plug into the same
runner and submission tooling.

Our job: **beat the FM baseline of primary 0.5946**, measured against the oracle
ceiling of 0.8645 (real headroom ≈ 0.27, not 0.41). See `task.md` for the
full goal, the metric's true range, and what's already been measured.

---

## 2. Directory map

```
Tippytop/
├── README.md                 quick start + rules
├── task.md                   goal, strategy, measured headroom (READ THIS)
├── techjam2026_..._agent.md  official problem statement (from the site)
├── docs/
│   ├── tutorial.md           how to get started + add a model
│   └── project-structure.md  ← this file
│
├── kuairand-starter-kit/     VENDORED — DO NOT EDIT (evaluate.py = frozen spec)
│
├── src/tippytop/             our package
│   ├── kit.py                bridge to the frozen kit (load/encode/evaluate)
│   ├── config.py             paths, baseline/oracle numbers, convergence rule
│   ├── data/
│   │   ├── features.py       feature engineering (within-user-varying signals)
│   │   └── sequences.py      per-user behaviour history (for DIN/SIM)
│   ├── losses/
│   │   └── ranking.py        pointwise / BPR / listwise objectives
│   ├── models/
│   │   ├── base.py           the Model contract every model implements
│   │   ├── __init__.py       model registry (@register)
│   │   └── fm.py             FM baseline adapter (+ your models beside it)
│   ├── training/runner.py    shared load→encode→fit→score loop
│   └── submission.py         build/validate submission CSVs
│
├── scripts/
│   ├── download_data.sh/.ps1 fetch KuaiRand-Pure into the kit
│   ├── run_experiment.py     train a model, print metrics vs baseline
│   └── make_submission.py    model → submission CSV → validate
│
├── experiments/configs/      one YAML per notable run (reproducibility)
├── results/
│   ├── leaderboard.md        shared scoreboard — log every run
│   └── submissions/          generated CSVs (git-ignored)
├── tests/test_harness.py     sanity: random ≈ 0.475
└── notebooks/                free-form exploration
```

---

## 3. The key design decisions (and why)

**a. The starter kit stays frozen and vendored.**
`kuairand-starter-kit/` is never modified. `evaluate.py` is the scoring spec —
keeping it pristine means we can always prove we didn't change how we're scored.
We reach it through `src/tippytop/kit.py`, which puts it on the path and
re-exports `load`, `encode`, `evaluate`. **Never import the kit any other way,
and never edit inside that folder.**

**b. One tiny model contract → no merge conflicts.**
Every model subclasses `Model` in `models/base.py` — just two methods:
`fit(enc, dim)` and `predict(X) → scores`. Each model lives in its **own file**
and registers itself by name (`@register("fm")`). Adding a model never touches a
shared file, so people work in parallel cleanly.

**c. Losses are separate from models.**
The #1 headroom (swap pointwise logloss → ranking loss) lives in
`losses/ranking.py`, independent of any model. That work stream doesn't need to
own a model to make progress.

**d. One shared runner + one leaderboard.**
`training/runner.py` is the single load→encode→fit→score path, so every result
is produced identically and is comparable. Log each run in
`results/leaderboard.md`.

**e. Data/features are a thin layer over the kit.**
`data/` wraps the kit's `load`/`encode`. Static-feature additions are known not
to help (measured) — the payoff is in signals that **vary within a user**
(watch time, engagement labels, sequences), which is why `features.py` and
`sequences.py` are their own files.

---

## 4. How to work day-to-day

1. **Set up once:** `pip install -r requirements.txt`, then
   `bash scripts/download_data.sh` (or the `.ps1`).
2. **Sanity check:** `python -m pytest tests/ -v` — `random` must score ≈ 0.475.
   If not, the harness is broken; fix that before trusting any result.
3. **Branch as `dev/<name>`** (e.g. `dev/alice`) and work in your files.
4. PR into `main`; put your best valid/test primary in the description.
5. **Run:** `python scripts/run_experiment.py --model <name>`.
6. **Measure on valid** while iterating. A change is real only if
   **Δ valid primary > +0.002** (that's the noise band; std is 0.0008).
7. **Log it** in `results/leaderboard.md` and save a config in
   `experiments/configs/`.

---

## 5. The rules (short version)

- ❌ Never edit `kuairand-starter-kit/`. Import via `tippytop.kit`.
- ✅ Report **valid** primary while iterating; **test** only for final picks.
- ✅ A change counts only if Δvalid > +0.002.
- ✅ Every run goes in `results/leaderboard.md`.
- ✅ Submissions: header `row_id,user_id,video_id,score`, exact eval-row order,
  validate with the kit's `submit.py --check` before submitting.

---

## 6. Where to read more

| Question | File |
|---|---|
| How do I actually get started / add a model? | `docs/tutorial.md` |
| What are we trying to do, and what's the strategy? | `task.md` |
| First tasks / ideas per person | `task.md` (headroom section) |
| The official problem statement | `techjam2026_track2_autonomous_ml_research_agent.md` |
| How to add a model | `src/tippytop/models/README.md` |
| Current scores | `results/leaderboard.md` |

> Status: the **baselines are done** — `fm` (reproduces 0.5946), `pop`, and
> `random` all run through the CLI, and submission write/check/score work. The
> stubs left to fill are the headroom directions: `losses/ranking.py`,
> `data/sequences.py`, `data/features.py`, and new models beside `models/fm.py`.
