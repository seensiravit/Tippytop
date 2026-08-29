# KuaiRand-Pure Starter Kit

## Dependencies

Python 3.9+ and numpy. **Nothing else.** No torch, pandas, or sklearn required.

## Data

Download from https://kuairand.com (direct Zenodo link, no registration needed):

```bash
# Run inside the Starter Kit directory; unpacking gives you ./KuaiRand-Pure/
wget https://zenodo.org/records/10439422/files/KuaiRand-Pure.tar.gz
tar xzf KuaiRand-Pure.tar.gz
```

## Running

```bash
python3 baseline.py --model fm
```

`--data_dir` defaults to `./KuaiRand-Pure/data`; specify it explicitly if the data lives elsewhere.

`--model` accepts `fm` (official baseline) / `pop` (trivial baseline) / `random` (lower bound, for sanity-checking the eval code).
FM takes about 40 seconds end-to-end (CPU, single core).

## Task definition (specification is fixed — do not change it)

| | |
|---|---|
| Task | **Within-user ranking** — each user only ranks their own impressions in the eval set; no full-corpus retrieval |
| Relevance label | `long_view` (native column, 0/1) |
| Metrics | `GAUC`, `nDCG@5`; **primary score = mean of the two** |
| Data split | train `20220408–20220421` / valid `20220422–20220428` / test `20220429–20220508` |
| Zero-positive users | nDCG scored as 0.0 and included in the mean; GAUC only counts users with `0 < positives < impressions`, weighted by positive count |
| nDCG gain | `2^rel − 1` (equivalent to identity under binary labels) |

Implementation is in `evaluate.py`; all conventions are documented in the file's header comment.

## Baseline ladder

Scores on the test set. **The row to beat is FM.**

| | GAUC | nDCG@5 | primary |
|---|---|---|---|
| random (lower bound, sanity check) | 0.4996 | 0.4511 | 0.4753 |
| item popularity (trivial) | 0.6308 | 0.5121 | 0.5715 |
| **FM (official baseline)** | **0.6610** | **0.5282** | **0.5946** |

### ⚠️ The real range of the metric: the ceiling for nDCG@5 is 0.729, not 1.0

Of the 23,875 users in the test set:

| | Share | Effect on the metric |
|---|---|---|
| All-negative users (none of their impressions are long_view) | **27.1%** | nDCG is always **0**; no model can fix this; excluded from GAUC |
| All-positive users | **9.2%** | nDCG is always **1**; excluded from GAUC |
| Discriminative users | **63.7%** | the actual sample GAUC is computed on |

So even using the true labels as prediction scores (oracle, perfect ranking) you can only reach:

| | random | FM baseline | **oracle ceiling** | fraction FM already captured |
|---|---|---|---|---|
| GAUC | 0.4996 | 0.6610 | **1.0000** | 32.3% |
| nDCG@5 | 0.4511 | 0.5282 | **0.7289** | 27.8% |
| **primary** | 0.4753 | **0.5946** | **0.8645** | **30.7%** |

**Measure your progress against the oracle as the denominator.** Seeing 0.5946 and thinking "still far from a perfect 1.0" is a misreading —
the baseline has already captured about a third of the usable range, and the remaining headroom is 0.27, not 0.41.

FM's std over 5 random seeds is **0.0008** on every metric. From this, the convergence criterion is **ε = 0.002 (≈2.5σ), N = 3**:
if the validation primary score improves by no more than 0.002 for 3 consecutive iterations, consider it converged.

> Sanity check: if running `--model random` through your eval code does not yield primary ≈ 0.475 (±0.001), your harness is broken — fix it first.

## Submission format

CSV with a header row, one line per row of the eval set:

```
row_id,user_id,video_id,score
0,0,7531,-3.34176
1,0,4214,-1.4955
...
```

| Field | Description |
|---|---|
| `row_id` | 0-based, consecutively increasing, matching the row order of `data.load()[split]` (deterministic: read `log_standard_4_08_to_4_21_pure.csv` first, then `log_standard_4_22_to_5_08_pure.csv`, then filter by date while preserving original file order) |
| `user_id` / `video_id` | redundant fields, used only to verify alignment |
| `score` | the score your model assigns to this row; any real number, only relative order matters; NaN / Inf not allowed |

> **Why `row_id` is mandatory:** `(user_id, video_id)` is **not unique** in the eval set —
> the test set has 3.06% duplicate pairs, repeated up to 12 times. So it cannot serve as a primary key.

Generate and validate:

```bash
python3 submit.py --make  --split test  submission.csv    # generate a sample submission from the official FM baseline
python3 submit.py --check --split test  submission.csv    # validate format and alignment
python3 submit.py --score --split valid submission.csv    # validate and score (local valid only)
```

`--check` rejects: wrong header, wrong row count, `row_id` gaps, `user_id`/`video_id` misaligned with the eval set,
and `score` that is non-numeric or NaN/Inf. **Run `--check` yourself before submitting.**

## Where to start improving

The ordering below is **measured, not guessed**. Dead ends the organizers already tried are marked so you don't repeat them.

### Already measured: these two yield nothing — don't waste iterations

| Tried | Result |
|---|---|
| **Adding static features** — wiring in all 13 of CWM's feature fields (+`music_id`/`video_type`/`upload_type` + 6 coarse user-side buckets) | primary **0.5940** vs **0.5950** with 5 fields — no difference within noise, if anything slightly worse |
| **Adding model capacity** — embedding dimension k = 8 / 16 / 32 | 0.5895 / 0.5902 / 0.5887, barely moves |

Reason: the `user_id × video_id` cross already captures most of the learnable signal. Coarse buckets like `follow_user_num_range`
are redundant given `user_id`; and 1.14M rows can't support larger capacity. **The bottleneck is not features or capacity.**

⚠️ Also note: **the first-order term of any pure user-side feature contributes exactly 0 to the score.** Because ranking is done
within each user, any term that is constant within a user does not change the intra-group order (measured: `item_pop × user bias`
and pure `item_pop` scored identically to the last digit). User-side features can only matter through **crosses with item-side features**.

### Unexplored: the headroom should be here

Ordered by our estimated likelihood of payoff (**the organizers did not test these — they are left for you**):

1. **Change the loss function.** It is currently pointwise logloss, but the metrics (GAUC / nDCG) are **ranking metrics**.
   Switch to pairwise (BPR) or listwise (softmax over each user's impressions) — aligning the objective with the eval spec.
   This is the one we think most likely to work.
2. **User history sequences.** The current features **make no use of behavior sequences**. Each KuaiRand user has hundreds to
   thousands of interactions in train; interest modeling of the DIN / SIM family is a completely open direction.
3. **Multi-objective.** The logs also have `is_click`, `is_like`, `is_follow`, `is_comment`, `is_forward`, `play_time_ms`,
   usable as auxiliary tasks alongside the `long_view` main task.
4. **Modeling watch time.** [CWM](https://github.com/hyz20/CWM)'s contribution is exactly this: it treats watch time as
   **censored regression** (when a video plays to the end, the true watch time is truncated, so it uses a one-sided loss
   instead of squared error). A direction with research depth.
5. **Change the model.** DeepFM / DCN / xDeepFM. Since capacity is measurably not the bottleneck, **rank this below 1–4.**
6. **Time features and distribution drift.** `hourmin`, `date`, and the drift between train and test.
7. **Unbiased validation (advanced).** `log_random_4_22_to_5_08_pure.csv` is a randomly-exposed log (1.18M rows),
   usable as an extra unbiased validation set to check whether the model overfits only to biased traffic.

## Using your own model (including CWM)

`evaluate.py` is fully decoupled from the model — it only needs three equal-length arrays:

```python
from evaluate import evaluate
print(evaluate(user_ids, labels, scores))   # scores can come from any model
```

- `user_ids`: the user_id of each row in the eval set
- `labels`: that row's `long_view` (0/1)
- `scores`: the score your model assigns to that row (any real number, only relative order matters)

So you can skip `baseline.py` entirely and use PyTorch, LightGBM, or CWM's xDeepFM — just hand the final
`scores` to `evaluate()`. **The scoring spec is defined solely by `evaluate.py`.**

> Note on using CWM: it depends on `torch==1.6.0` (a 2020 version, likely won't install on newer GPUs),
> its loss optimizes counterfactual watch time, and its eval label is a self-reconstructed `long_view2`.
> It is research code for a watch-time debiasing paper — useful as an **advanced reference**, not recommended as a starting point.

## Files

| | |
|---|---|
| `evaluate.py` | metric implementation + all spec conventions. **Do not change.** |
| `data.py` | data loading, official split, feature encoding. Add features here. |
| `baseline.py` | the three baselines. FM is the one to beat. |
| `baseline_scores.json` | official published scores + seed variance + convergence parameters. |
| `submit.py` | generate / validate submission files. |
| `ablation_features.py` | feature ablation experiment; reproduces the "adding features yields nothing" numbers. |
