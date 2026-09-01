# Final run artifacts

**This directory is exempt from `.gitignore` on purpose.** Everything else the
agent writes is scratch — `setup` archives the previous run's copies on every
new tag — but these files are graded deliverables and must be committed.

**Populated with the submission run** (5 iterations, valid 0.6045, test 0.5976,
0 manual interventions, converged by plateau rule). See `resource_report.json`
for full resource usage.

## What goes here

| File | Deliverable | Contents |
|---|---|---|
| `runs.jsonl` | 3 — run & iteration logs | Per iteration: hypothesis, files changed, resulting metrics, outcome, mode, any error/recovery event, tokens, wall-clock |
| `resource_report.json` | 4 — results summary | Totals: iterations used, elapsed seconds, tokens in/out, concepts tried and confirmed, best valid/test primary |
| `submission.csv` | 4 — final model output | `row_id,user_id,video_id,score`, one row per test-split row |
| `results.tsv` | supporting | Per-experiment scoreboard |
| `concepts.json` | supporting | Concept lifecycle, including why each was closed (`expanded (maxed out…)`, `pivoted (no improvement)`) — this is the Autonomy evidence |

## How to fill it

One command, from the repo root, after the run you intend to submit:

```bash
python scripts/package_final_run.py            # --dry-run to check without copying
git add results/final_run && git commit -m "Add final run artifacts"
```

The script gates before it copies. It **refuses** when:

- any of the five required artifacts is missing (with the command that produces
  them — they only appear when `finalize` fires on convergence);
- `submit.py --check --split test` rejects the submission. This is the single
  largest avoidable risk in the whole entry: a malformed or misaligned CSV scores
  zero regardless of model quality, and `(user_id, video_id)` is *not* a key on
  test — 3.06% of its rows are repeated pairs, so alignment is verified row by
  row against `row_id`;
- the run is short enough to be a smoke test (under 5 iterations);
- `resource_report.json` has no `manual_interventions` field, or reports a count
  that disagrees with `interventions.jsonl`. A wrong number is worse than none.

It also prints the iteration count, the outcome mix, wall clock, token totals and
the intervention summary — the numbers Deliverable 4 asks for and the ones the
Devpost draft in `docs/devpost.md` has bracketed placeholders for.

## Also required, and not produced automatically

- **The prose around the intervention count.** The number, the summary and every
  reason are written into `resource_report.json` automatically — resumes are
  detected whether or not the operator declares them, and
  `python -m autoresearch_lg.cli note "<reason>"` records the rest. What still
  needs a human is the explanation in the Devpost description of what each
  intervention was and why.
- **A results table** giving validation-best GAUC / nDCG@5 and the absolute delta
  over the official baseline. Numbers are in `../leaderboard.md`; the table is
  already drafted in `../../docs/devpost.md`.
- **GPU-hours**, if any were used. None so far — this is a CPU-only pipeline.

## Reporting note

Report progress against the **oracle ceiling of 0.8645**, not 1.0. About 36% of
users are unrankable by construction (27.1% all-negative, nDCG pinned to 0;
9.2% all-positive, pinned to 1), so the attainable range is much narrower than
it looks and the baseline already captures ~31% of it.
