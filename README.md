# Tippytop — TikTok TechJam 2026

Within-user ranking on **KuaiRand-Pure**. Goal: **beat the FM baseline (primary 0.5946)**;
measure progress against the oracle ceiling (0.8645). Full brief in [`task.md`](task.md).

**New here? Start with [`docs/tutorial.md`](docs/tutorial.md).**

## Quick start
```bash
pip install -r requirements.txt
# download data into the vendored kit:
bash scripts/download_data.sh          # or: powershell scripts/download_data.ps1
python -m pytest tests/ -v             # sanity: random ~= 0.475
python scripts/run_experiment.py --model fm
```

## Layout
```
Tippytop/
├── task.md                     our goal + strategy (measured headroom)
├── techjam2026_..._agent.md    official problem statement (from the site)
├── kuairand-starter-kit/       VENDORED, UNTOUCHED — evaluate.py is the frozen spec
├── src/tippytop/               our package (wraps the kit)
│   ├── kit.py                  bridge to the frozen kit (load/encode/evaluate)
│   ├── config.py               paths, baselines, convergence constants
│   ├── data/                   features.py, sequences.py (our feature layer)
│   ├── losses/                 ranking.py — BPR / listwise (top headroom)
│   ├── models/                 one file per model + registry (base.py contract)
│   ├── training/runner.py      shared load→encode→fit→score loop
│   └── submission.py           build/validate submission CSVs
├── scripts/                    run_experiment.py, make_submission.py, download_data.*
├── experiments/configs/        one YAML per run (reproducibility)
├── results/                    leaderboard.md + submissions/
├── tests/                      harness sanity checks
├── notebooks/                  exploration
└── docs/                       tutorial + structure guide
```

## Rules
- **Never edit `kuairand-starter-kit/`** — import it via `tippytop.kit`.
- Report **valid** primary while iterating; a change is real only if Δvalid > +0.002.
- Each person works on their own branch `dev/<name>`; PR into `main`.
- Log every run in `results/leaderboard.md`. See [`docs/tutorial.md`](docs/tutorial.md) to get started.
