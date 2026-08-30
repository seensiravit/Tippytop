# Final run artifacts

**This directory is exempt from `.gitignore` on purpose.** Everything else the
agent writes is scratch — `setup` archives the previous run's copies on every
new tag — but these files are graded deliverables and must be committed.

Empty until the submission run is chosen. Do not fill it with smoke-test output.

## What goes here

| File | Deliverable | Contents |
|---|---|---|
| `runs.jsonl` | 3 — run & iteration logs | Per iteration: hypothesis, files changed, resulting metrics, outcome, mode, any error/recovery event, tokens, wall-clock |
| `resource_report.json` | 4 — results summary | Totals: iterations used, elapsed seconds, tokens in/out, concepts tried and confirmed, best valid/test primary |
| `submission.csv` | 4 — final model output | `row_id,user_id,video_id,score`, one row per test-split row |
| `results.tsv` | supporting | Per-experiment scoreboard |
| `concepts.json` | supporting | Concept lifecycle, including why each was closed (`expanded (maxed out…)`, `pivoted (no improvement)`) — this is the Autonomy evidence |

## How to fill it

After the run you intend to submit, from the repo root:

```bash
cp runs.jsonl resource_report.json submission.csv results.tsv concepts.json results/final_run/
uv run python submit.py --check --split test results/final_run/submission.csv
git add results/final_run && git commit -m "Add final run artifacts"
```

Run the checker **before** committing. It rejects a wrong header, a row-count
mismatch, `row_id` gaps, misalignment against the evaluation split, and
non-numeric or NaN/Inf scores — a malformed file scores zero regardless of model
quality.

## Also required, and not produced automatically

- **Manual intervention count.** Deliverable 3 asks for it explicitly, and
  Impact & Relevance (20%) is scored primarily on this number. Record it here,
  with what each intervention was and why. An honest count beats an unsupported
  claim of full autonomy.
- **A results table** giving validation-best GAUC / nDCG@5 and the absolute delta
  over the official baseline. Numbers are in `../leaderboard.md`.
- **GPU-hours**, if any were used. None so far — this is a CPU-only pipeline.

## Reporting note

Report progress against the **oracle ceiling of 0.8645**, not 1.0. About 36% of
users are unrankable by construction (27.1% all-negative, nDCG pinned to 0;
9.2% all-positive, pinned to 1), so the attainable range is much narrower than
it looks and the baseline already captures ~31% of it.
