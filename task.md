# Problem Statement — Tippytop (TikTok TechJam 2026)

## What we are building

A **short-video recommendation ranker** on the **KuaiRand-Pure** dataset (Kuaishou logged
interactions). Given the impressions shown to a user, we must **order that user's own impressions**
so the videos they will genuinely engage with rank highest.

## The task (fixed spec — defined solely by `evaluate.py`, do not change it)

| | |
|---|---|
| **Task type** | **Within-user ranking** — each user ranks only their own impressions in the eval set. No cross-user comparison, no full-corpus retrieval. |
| **Label** | `long_view` (native 0/1 column) = the user watched the video long enough to count as real engagement. |
| **Metrics** | `GAUC` and `nDCG@5`; **primary score = mean of the two**. |
| **Data split** | train `20220408–20220421` / valid `20220422–20220428` / test `20220429–20220508`. |
| **Scoring input** | three equal-length arrays: `user_ids`, `labels`, `scores`. The model is fully decoupled from scoring — any model works as long as it emits a `scores` array in eval-row order. |

Only the **relative order** of scores within a user matters — never absolute values, never comparisons across users.

## Our goal

**Beat the FM baseline of `primary = 0.5946`** on the test set.

| Model | GAUC | nDCG@5 | primary |
|---|---|---|---|
| random (sanity floor) | 0.4996 | 0.4511 | 0.4753 |
| item popularity (trivial) | 0.6308 | 0.5121 | 0.5715 |
| **FM (baseline — the row to beat)** | **0.6610** | **0.5282** | **0.5946** |
| **oracle ceiling (perfect ranking)** | 1.0000 | 0.7289 | **0.8645** |

### Read the metric correctly
The ceiling is **0.8645, not 1.0**. 27.1% of test users are all-negative (nDCG always 0, unfixable);
9.2% are all-positive (nDCG always 1). Only **63.7%** of users are discriminative. So:
- FM has already captured ~30.7% of the usable range.
- The real remaining headroom is **~0.27** (0.5946 → 0.8645), not 0.41.
- **Measure progress against the oracle (0.8645) as the denominator**, not against 1.0.

### Convergence criterion
FM's std over 5 seeds is 0.0008 on every metric. Consider a change converged if the **validation**
primary improves by **≤ 0.002 (≈2.5σ) for 3 consecutive iterations** (ε = 0.002, N = 3).

## Constraints & ground rules

- **Runs on CPU, numpy only** — no torch/pandas/sklearn required for the baseline (Python 3.9+).
- **`evaluate.py` is frozen.** All spec conventions live there. Do not edit it.
- **Sanity check first:** `--model random` must score primary ≈ 0.475 (±0.001). If not, the harness is broken — fix that before trusting any result.
- **Submission:** CSV with header `row_id,user_id,video_id,score`, one row per eval row, in the exact
  order of `data.load()[split]` (read `log_standard_4_08_to_4_21_pure.csv`, then
  `log_standard_4_22_to_5_08_pure.csv`, then filter by date preserving file order). `row_id` is
  mandatory because `(user_id, video_id)` is **not** unique (3.06% duplicate pairs in test). Validate
  with `python3 submit.py --check` before submitting.

## What does NOT work (already measured — don't repeat)

1. **Adding static features** (video-side or user-side): 0.5940 vs 0.5950 — no change within noise.
2. **Adding model capacity** (embedding dim k = 8/16/32): 0.5895 / 0.5902 / 0.5887 — flat.
3. **Pure user-side features contribute exactly 0** — ranking is within-user, so any feature constant
   across a user's impressions cannot change their order. User attributes matter **only through crosses
   with item-side features**.

> The bottleneck is **not features and not capacity.** It is the **objective** and the **unused signals.**

## Where the headroom is (our strategy, ranked by expected payoff)

1. **Change the loss → ranking objective.** Baseline trains pointwise logloss but is scored on ranking
   metrics (GAUC/nDCG). Switch to **pairwise (BPR)** or **listwise (per-user softmax)**. Highest-value
   single change, needs no new data. **Start here.**
2. **User behavior sequences.** Each user has hundreds–thousands of train interactions, currently unused.
   Interest modeling (DIN / SIM family).
3. **Multi-objective learning.** Use `is_click`, `is_like`, `is_follow`, `is_comment`, `is_forward`,
   `play_time_ms` as auxiliary tasks alongside the `long_view` main task.
4. **Watch-time modeling.** Treat `play_time_ms` as censored regression (CWM-style one-sided loss).
5. **Different model** (DeepFM / DCN / xDeepFM) — ranked below, since capacity is not the bottleneck.
6. **Time features & drift** (`hourmin`, `date`, train↔test distribution shift).
7. **Unbiased validation (advanced):** use `log_random_4_22_to_5_08_pure.csv` (randomly-exposed log) to
   check overfitting to biased traffic.

## Definition of done

- [ ] Data downloaded; `--model random` reproduces primary ≈ 0.475 (harness verified).
- [ ] FM baseline reproduced at primary ≈ 0.5946.
- [ ] At least one headroom direction (starting with the ranking loss) implemented and measured on **valid**.
- [ ] A model that beats **valid** primary of the FM baseline beyond the noise band (Δ > 0.002).
- [ ] Valid submission CSV generated and passing `submit.py --check`.
- [ ] Final test-set primary reported vs. the 0.5946 baseline and the 0.8645 oracle ceiling.
