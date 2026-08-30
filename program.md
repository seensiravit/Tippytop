# autoresearch — KuaiRand-Pure

This is an experiment to have an LLM agent do its own recommendation-modeling
research, autonomously, overnight. It adapts the loop from
[autoresearch-win-rtx](https://github.com/jsegov/autoresearch-win-rtx)
(itself a fork of [karpathy/autoresearch](https://github.com/karpathy/autoresearch),
which trains nanochat on a fixed GPU time budget) to this repo's task instead:
**within-user ranking on KuaiRand-Pure**, scored by `evaluate.py`.

The mapping, if you know the original:

| autoresearch-win-rtx | here |
|---|---|
| `prepare.py` (fixed, read-only) | `evaluate.py` (fixed, read-only — its own header says "口径全部写死在这里，不要改") |
| `train.py` (the file you edit) | `baseline.py` + `data.py` (model, loss, training loop, features) |
| `val_bpb`, lower is better | `primary` = mean(GAUC, nDCG@5) on **valid**, higher is better |
| fixed 5-minute GPU wall clock | no hard GPU clock (CPU/numpy); see the runtime ceiling below instead |
| VRAM soft constraint | none — instead: **no new dependencies** (numpy + stdlib only) |

## Setup

Work with the human to:

1. **Agree on a run tag**: propose one based on today's date (e.g. `aug28`). The
   branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current `master`/`main`.
3. **Read the in-scope files**:
   - `README.md` — task definition, scoring convention, the two already-tested
     dead ends, and the ranked list of unexplored directions. Read this fully
     before proposing ideas — do not re-discover what's already been ruled out.
   - `evaluate.py` — the ground truth metric. **Do not modify.**
   - `data.py` — data loading, official splits, feature encoding. Editable:
     this is where new features/fields go.
   - `baseline.py` — the three baselines. `run_fm`/`FM` is the file you'll
     spend most of your time changing (loss, optimizer, architecture,
     training loop), but you may also add whole new model classes here or in
     new files.
   - `baseline_scores.json` — official numbers, per-seed std, and the
     convergence rule (`epsilon=0.002, N=3`). This std is your noise floor.
4. **Verify data exists**: check that `./KuaiRand-Pure/data/` contains the six
   CSVs (`video_features_basic_pure.csv`, `video_features_statistic_pure.csv`,
   `user_features_pure.csv`, `log_standard_4_08_to_4_21_pure.csv`,
   `log_standard_4_22_to_5_08_pure.csv`, `log_random_4_22_to_5_08_pure.csv`).
   If not, tell the human to follow the download steps in `README.md`.
5. **Initialize `results.tsv`**: create it with just the header row (see
   Logging below). The baseline gets recorded after the first run.
6. **Confirm and go.**

Once confirmed, kick off the experimentation loop.

## Experimentation

Each experiment is a single `python3 baseline.py ...` invocation (or your own
entrypoint, if you add one — keep it a single command that loads data, trains,
and prints a summary).

**Runtime ceiling**: data loading takes ~5s, the FM baseline trains in
~30-40s, so a full run finishes in well under a minute on CPU. That headroom
is deliberate — use it. Budget **up to 5 minutes wall-clock per experiment**
(more epochs, bigger embeddings, a sequence model). If a run exceeds **10
minutes**, kill it and treat it as a failure (discard).

**What you CAN do:**
- Modify `baseline.py` — loss function (pointwise → pairwise/BPR/listwise),
  optimizer, training loop, model architecture, or add new model classes
  entirely (DIN/SIM-style sequence models, multi-task heads, censored
  regression on watch time, DeepFM/DCN, etc).
- Modify `data.py` — add fields to `FIELDS`, change bucketing, build user
  history sequences, pull in `is_click`/`is_like`/`play_time_ms`/etc for
  auxiliary tasks, use `log_random_4_22_to_5_08_pure.csv` as an unbiased
  validation set.

**What you CANNOT do:**
- Modify `evaluate.py`, the `SPLITS` dates, or `LABEL` in `data.py`. The
  scoring convention is fixed by the task, not by you.
- Install new packages. Numpy + stdlib only — same constraint the whole kit
  already runs under. If an idea genuinely needs something else (e.g. torch
  for a sequence model), stop and ask the human first; don't `pip install`
  autonomously.
- Use **test-split** labels to make any keep/discard decision. `test` is
  measured for reporting only, exactly like a real holdout. All tuning
  decisions come from `valid`.

**The goal**: maximize `valid` primary. Track progress against the oracle
ceiling in `baseline_scores.json` (0.8484 valid / 0.8645 test), not against
1.0 — the README explains why 1.0 is unreachable (all-negative/all-positive
users).

**Start here, not from scratch** — `README.md`'s "从哪里开始改" section is a
ranked, already-reasoned list of directions the maintainers have *not* tried
(pairwise/listwise loss is their top pick), plus two directions they *have*
tried with **no gain** (more static features, more FM capacity via larger
`k`). Do not re-run those two; start from the ranked list instead, and only
go beyond it once those are exhausted.

**Simplicity criterion**: all else equal, simpler is better. A small
improvement that adds ugly complexity is not worth it. An equal-or-better
result from *removing* complexity is a great outcome — keep it. When
evaluating a change, weigh the complexity cost against the improvement
against the noise floor (below).

**Noise floor**: `baseline_scores.json` reports std ≈ 0.0008 on test primary
across 5 seeds for the FM baseline. Treat any single-seed valid-primary
delta **smaller than ~0.002** as noise, not signal — matches the project's
own documented convergence rule (`epsilon=0.002`). For a change that looks
like a real but marginal win (0.002–0.004), rerun it with 1-2 more seeds
before committing to "keep" — don't advance the branch on a single lucky
seed.

**The first run**: your very first run is the baseline as-is
(`python3 baseline.py --model fm --seed 0`), to reconfirm the numbers you
inherited (valid primary ≈0.6015, test primary ≈0.5946-0.5953).

## Output format

`baseline.py --model fm` prints per-epoch progress, then a summary:

```
=== fm (seed=0) ===
  valid  GAUC 0.6671 | nDCG@5 0.5358 | primary 0.6015
  test   GAUC 0.6621 | nDCG@5 0.5286 | primary 0.5953
```

Extract the summary lines with:

```
grep -A2 "^=== " run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT
comma — commas break in free-text descriptions).

Header and columns:

```
commit	valid_primary	test_primary	wall_seconds	status	description
```

1. git commit hash (short, 7 chars)
2. `valid` primary achieved (e.g. `0.601500`) — `0.000000` for crashes
3. `test` primary, recorded for reference only, never used to decide keep/discard
4. wall-clock seconds for the run, rounded to 1 decimal
5. status: `keep`, `discard`, or `crash`
6. short free-text description of what this experiment tried

Example:

```
commit	valid_primary	test_primary	wall_seconds	status	description
23ab391	0.601500	0.595300	38.4	keep	baseline (FM, k=16)
b2c3d4e	0.604800	0.598100	41.2	keep	pairwise BPR loss instead of pointwise logloss
c3d4e5f	0.600200	0.594000	39.0	discard	larger k=32 (already ruled out, resample-checked)
d4e5f6g	0.000000	0.000000	0.0	crash	user history sequence encoder (shape mismatch)
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/aug28`).

LOOP FOREVER:

1. Look at the git state: current branch/commit.
2. Tune `data.py`/`baseline.py` with one experimental idea by directly
   editing the code.
3. `git commit`.
4. Run the experiment: `python3 baseline.py --model fm [...] > run.log 2>&1`
   (redirect everything — do not let output flood your context).
5. Read the results: `grep -A2 "^=== " run.log`.
6. If that output is empty, the run crashed. `tail -n 50 run.log` for the
   traceback and attempt a fix if it's something small (typo, shape
   mismatch). If you can't get it working after a few attempts, give up on
   the idea.
7. Record the results in `results.tsv`.
8. If `valid` primary improved by more than the noise floor (~0.002; rerun
   1-2 more seeds first if it's a marginal 0.002-0.004 win) — **keep**,
   advance the branch.
9. Otherwise — **discard**, `git reset --hard` back to where this experiment
   started.

You are a completely autonomous researcher. If an idea works, keep it and
advance the branch so the next idea builds on it. If it doesn't, discard and
try the next one. Rewinding past "the last kept commit" should be very rare.

**Crashes**: if a run crashes, use judgment — fix small bugs and re-run; if
the idea itself is fundamentally broken, log `crash` in `results.tsv` and
move to the next idea.

**Timeout**: budget ~5 minutes total per experiment (a few seconds startup +
train). If a run exceeds 10 minutes, kill it, log it as a failure, and
discard.

**NEVER STOP**: once the loop has begun (after setup), do not pause to ask
the human whether to continue. Do not ask "should I keep going?" The human
may be asleep or away and expects you to work *indefinitely* until manually
stopped. If you run out of ideas from the README's ranked list, think
harder — re-read `data.py`/`baseline.py` for angles not yet tried, combine
previous near-misses, or go beyond the ranked list (e.g. DeepFM/DCN/xDeepFM,
per the README's own suggestion once 1-4 are exhausted). The loop runs until
the human interrupts you, period.

At ~40 seconds/experiment (vs. the original repo's 5 minutes/experiment on
GPU), you can run far more than the ~12/hour the original was designed
around — budget accordingly, but don't let raw iteration count substitute for
actually reading why an idea did or didn't work before moving to the next
one.
