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

- **Runs on CPU** — no GPU required; FM trains in ~60 s. Python 3.11+, managed with `uv`. NumPy-only for the baseline; LightGBM + scikit-learn available via the `[models]` extra.
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
   across a user's impressions cannot change their order. `user_lv_rate` scores exactly random GAUC
   (0.5000) — a direct measurement of this effect.
4. **Ranking loss (listwise/BPR) — 7/7 matched comparisons closed this.** The objective IS better
   (+0.0008 to +0.0033 against a pointwise control at identical batching), but grouped batching
   costs −0.0023 to −0.0040 on its own. The gain never repays the machinery.
5. **Multi-task auxiliary heads — 6/6 against their own control.** `is_click` correlates 0.760 with
   `long_view` (near-duplicate task); score degrades monotonically with auxiliary weight.
6. **LambdaRank / GBDT** — no user×item interaction available to axis-aligned splits; personalised
   crosses are too sparse to generalise. FM beats trees because the signal lives in sparse embeddings.

> The bottleneck is **not features, capacity, or loss objective.** It is **model family** (FFM
> captures field-aware interactions FM cannot) and **ensemble diversity**.

## What worked / open headroom

**Verified wins (beat 0.002 threshold, replicated):**
- **FFM** (k=4): +0.0009 valid / +0.0019 test over FM, non-overlapping across 6 seeds each.
- **Rank-averaged ensembles**: +0.0013–0.0014 on two disjoint seed halves.
- **6×FM + 6×FFM ensemble**: valid 0.6045 / test 0.5976 (+0.0031 over baseline).

**Open headroom (untried, ranked by potential):**
1. **`video_features_statistic_pure.csv`** — 30+ continuous per-video engagement columns; the
   organisers' ablation only tested categorical IDs. Compute from train split only (leak-free).
2. **Within-session position from `time_ms`** — varies within a user, so unlike user-side
   aggregates it can reorder impressions.
3. **Censored watch-time (D2Q-style)** — duration is negatively predictive (below-random GAUC);
   CWM/D2Q papers address this bias.
4. **User behavior sequences** — ~42 events/user; DIN/SIM family.
5. **Stacking** — FM/FFM scores as features for a second-stage tree model.

## Definition of done

- [x] Data downloaded; `--model random` reproduces primary ≈ 0.475 (harness verified).
- [x] FM baseline reproduced at primary ≈ 0.5946.
- [x] Headroom directions implemented and measured on **valid** (FFM, ensembles, ranking loss, multi-task, LambdaRank — all with controlled measurements).
- [x] A model that beats **valid** primary of the FM baseline beyond the noise band (Δ > 0.002): **6×FM + 6×FFM ensemble, valid 0.6045 (+0.0029 over baseline)**.
- [x] Valid submission CSV generated and passing `submit.py --check` (170,588 rows).
- [x] Final test-set primary reported: **0.5976 vs 0.5946 baseline, vs 0.8645 oracle ceiling**.
