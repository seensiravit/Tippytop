# Data & Features

This document describes the dataset behind the starter kit: where it comes from, which files
and columns exist, what the current code actually consumes, and what is left on the table.

> **Note:** the raw data is **not** shipped with the kit — you download it separately (see below).
> Column lists here follow the KuaiRand-Pure schema; the "currently used" markers are read
> directly from `data.py` and `ablation_features.py`. Verify exact column names against the CSV
> headers once you've downloaded the data, as minor naming can vary between KuaiRand releases.

---

## 1. The dataset: KuaiRand-Pure

- **Source:** [KuaiRand](https://kuairand.com) — an unbiased sequential-recommendation dataset
  released by Kuaishou (the short-video platform), from the paper
  *"KuaiRand: An Unbiased Sequential Recommendation Dataset with Randomly Exposed Videos"* (CIKM 2022).
- **Variant:** *Pure* — the smallest of the three KuaiRand variants (Pure / 1K / 27K), chosen so the
  whole pipeline runs on a CPU in numpy with no heavy dependencies.
- **Download** (direct Zenodo link, no registration):
  ```bash
  wget https://zenodo.org/records/10439422/files/KuaiRand-Pure.tar.gz
  tar xzf KuaiRand-Pure.tar.gz          # → ./KuaiRand-Pure/data/
  ```
- **What it is:** logged interactions between users and short videos on the Kuaishou feed. Each row
  is one **impression** — a video that was shown to a user — together with how the user reacted
  (clicked, liked, watched-long, watch time, etc.).
- **Time span:** 2022-04-08 to 2022-05-08 (one month), which the kit splits into train / valid / test.

### Approximate scale

| | |
|---|---|
| Users | ~27K |
| Videos | ~7.5K |
| Standard-log interactions | ~1.4M rows |
| Train rows (04-08 → 04-21) | ~1.14M (per README) |
| Test users | 23,875 (per `baseline_scores.json`) |

The unique thing about KuaiRand is the **randomly-exposed** log: alongside the normal
recommender-driven feed, some videos were shown uniformly at random, giving an *unbiased* slice of
data useful for evaluation without the recommender's own selection bias.

---

## 2. Files in `KuaiRand-Pure/data/`

| File | Role | Used by the kit? |
|---|---|---|
| `log_standard_4_08_to_4_21_pure.csv` | Standard (recommender-driven) interaction log, first two weeks | ✅ read in `data.load()` — becomes **train** |
| `log_standard_4_22_to_5_08_pure.csv` | Standard interaction log, last ~two weeks | ✅ read in `data.load()` — becomes **valid** + **test** |
| `log_random_4_22_to_5_08_pure.csv` | **Randomly-exposed** interaction log (~1.18M rows), unbiased | ❌ not used — suggested for unbiased validation (README §7) |
| `video_features_basic_pure.csv` | Per-video metadata (author, type, music, duration…) | ✅ only `author_id` is read in `data.load()`; more fields read in `ablation_features.py` |
| `video_features_statistic_pure.csv` | Per-video aggregate statistics (counts of plays, likes…) | ❌ not used |
| `user_features_pure.csv` | Per-user profile (activity level, follower buckets, one-hot features…) | ❌ not used in the baseline; read in `ablation_features.py` |

The split by date happens in `data.py`:

```python
SPLITS = {'train': (20220408, 20220421),
          'valid': (20220422, 20220428),
          'test':  (20220429, 20220508)}
```

Note both standard log files are concatenated first, **then** filtered by date — so valid and test
both come out of the second file, and train comes from the first.

---

## 3. The label: `long_view`

The prediction target is **`long_view`** — a binary (0/1) column native to the log.

- `1` = the user watched the video "long enough" (Kuaishou's own long-view definition, a proxy for
  genuine engagement / satisfaction rather than an accidental tap).
- `0` = otherwise.

In `data.py` it is binarized defensively as `1 if r['long_view'] != '0' else 0`.

This is the **only** signal the metric scores against (`evaluate.py`). Everything else in the log is
a *potential feature*, not the target.

---

## 4. Interaction-log columns (each impression)

These columns live in the `log_*` CSVs. One row = one video shown to one user.

| Column | Meaning | Used now? |
|---|---|---|
| `user_id` | Which user saw the impression | ✅ **feature + eval grouping key** |
| `video_id` | Which video was shown | ✅ **feature** |
| `date` | Day (e.g. `20220408`) | ✅ used only to split train/valid/test |
| `hourmin` | Time of day (HHMM) | ❌ available (README §6: time features) |
| `time_ms` | Full timestamp of the impression | ❌ available |
| `tab` | Which UI surface/feed the impression came from | ✅ **feature** |
| `duration_ms` | Length of the video itself (ms) | ✅ **feature**, bucketed into 10 quantiles (`dur_bucket`) |
| `play_time_ms` | How long the user actually watched (ms) | ❌ available (README §3/§4: watch-time modeling) |
| **`long_view`** | **Watched long enough (0/1)** | ✅ **LABEL** |
| `is_click` | User clicked/entered the video | ❌ available (multi-task, README §3) |
| `is_like` | User liked it | ❌ available (multi-task) |
| `is_follow` | User followed the author from here | ❌ available (multi-task) |
| `is_comment` | User commented | ❌ available (multi-task) |
| `is_forward` | User forwarded/shared | ❌ available (multi-task) |
| `is_hate` | User disliked / hit "not interested" | ❌ available |
| `is_profile_enter` | User opened the author's profile | ❌ available |
| `profile_stay_time` | Time spent on the profile | ❌ available |
| `comment_stay_time` | Time spent in comments | ❌ available |
| `is_rand` | Was this a randomly-exposed impression (unbiased flag) | ❌ available |

> Exact column set can vary slightly by KuaiRand release; treat the `is_*` / `*_time` engagement
> columns as "present in the standard log" and confirm names against the header.

---

## 5. Video-side features (`video_features_basic_pure.csv`)

Keyed by `video_id`.

| Column | Meaning | Used now? |
|---|---|---|
| `video_id` | Join key | ✅ (join key) |
| `author_id` | Creator of the video | ✅ **feature** (in baseline) |
| `video_type` | e.g. normal vs. ad / short vs. long format | ⚠️ ablation only |
| `upload_type` | How the video was uploaded | ⚠️ ablation only |
| `music_id` | Background music track id | ⚠️ ablation only |
| `music_type` | Music category | ❌ available |
| `tag` | Content tag(s) / category | ❌ available |
| `upload_dt` | Upload date | ❌ available |
| `video_duration` | Duration (also present as `duration_ms` in the log) | ❌ available |
| `visible_status` | Visibility/moderation status | ❌ available |
| `server_width` / `server_height` | Video resolution | ❌ available |

"⚠️ ablation only" = wired in by `ablation_features.py` to test whether they help — the README
reports they **don't** move the score (0.5940 vs 0.5950, within noise).

There is also a `video_features_statistic_pure.csv` with aggregate play/like/finish counts per
video — untouched by the kit, a candidate for popularity-style item features.

---

## 6. User-side features (`user_features_pure.csv`)

Keyed by `user_id`. **Not used by the baseline** — the README explains why (see §8 below).

| Column | Meaning | Used now? |
|---|---|---|
| `user_id` | Join key | — |
| `user_active_degree` | Activity level bucket (e.g. high/middle/low active) | ⚠️ ablation only |
| `follow_user_num` / `follow_user_num_range` | # accounts the user follows (raw + coarse bucket) | ⚠️ range in ablation |
| `fans_user_num` / `fans_user_num_range` | # followers (raw + bucket) | ⚠️ range in ablation |
| `friend_user_num` / `friend_user_num_range` | # mutual friends (raw + bucket) | ⚠️ range in ablation |
| `register_days` / `register_days_range` | Account age (raw + bucket) | ⚠️ range in ablation |
| `is_lowactive_period` | Whether currently in a low-activity phase | ❌ available |
| `is_live_streamer` | User streams live | ❌ available |
| `is_video_author` | User uploads videos | ❌ available |
| `onehot_feat0` … `onehot_feat17` | 18 anonymized one-hot profile attributes | ❌ available |

The five range/degree buckets (`follow_user_num_range`, `register_days_range`,
`fans_user_num_range`, `friend_user_num_range`, `user_active_degree`) are the ones
`ablation_features.py` tries as CWM's user-side set.

---

## 7. What the baseline actually feeds the model

Only **5 fields** reach the FM, defined in `data.py`:

```python
FIELDS = ['user_id', 'video_id', 'author_id', 'tab', 'dur_bucket']
```

| Field | Origin | Type after encoding |
|---|---|---|
| `user_id` | log | categorical id |
| `video_id` | log | categorical id |
| `author_id` | `video_features_basic` (joined on `video_id`) | categorical id |
| `tab` | log | categorical id |
| `dur_bucket` | `duration_ms` → 10 quantile buckets | categorical id |

**Encoding** (`data.encode`): every field's raw values are mapped to contiguous integer ids from the
**train** vocabulary; each field reserves one extra `UNK` slot for values unseen in train (so valid/test
never crash on a new id). All fields share one flat embedding table via per-field `offsets`. The
output per split is `(X, y, users)` where `X` is `int32 (N, 5)`.

**Bucketing** (`data._bucket_edges`): `duration_ms` is cut at its train-set deciles (quantiles), so
`dur_bucket ∈ {0..9}`. Quantile edges (not fixed-width) keep the buckets roughly balanced.

---

## 8. What's used vs. available — the headroom map

| Category | Used by baseline | Available but unused |
|---|---|---|
| **Identity** | `user_id`, `video_id`, `author_id` | — |
| **Context** | `tab`, `dur_bucket` (from `duration_ms`) | `hourmin`, `date`, `time_ms` (time / drift) |
| **Video meta** | `author_id` | `music_id`, `video_type`, `upload_type`, `tag`, resolution, video statistics |
| **User profile** | *(none)* | activity degree, follower/following/friend buckets, register days, 18 one-hot feats |
| **Other engagement signals** | *(none — only `long_view` as label)* | `is_click/like/follow/comment/forward`, `play_time_ms`, profile/comment stay times |
| **Behavior sequences** | *(none)* | per-user interaction history (hundreds–thousands of rows) is fully unused |
| **Unbiased data** | *(none)* | `log_random_*` randomly-exposed log |

Two measured facts from the README worth repeating, because they shape which of the above is worth
your time:

1. **Adding static features (video-side or user-side) does not help.** The `user_id × video_id`
   embedding cross already captures most of the learnable signal, and 1.14M rows won't support more.
2. **Pure user-side features contribute exactly 0.** Ranking is *within-user*, so any feature that is
   constant across a user's own impressions cannot change their relative order. User attributes can
   only matter through **crosses with item-side features** — not on their own.

So the real headroom is in the columns the baseline throws away that vary *within a user*:
**behavior sequences** (§4 engagement columns over time), **watch time** (`play_time_ms`), and the
**other engagement labels** for multi-task learning — plus changing the *loss* to a ranking objective,
which needs no new data at all.

---

## 9. Quick reference: loading the data yourself

```python
from data import load, encode, FIELDS

splits = load('./KuaiRand-Pure/data')      # {'train': [...], 'valid': [...], 'test': [...]}
# each row is a tuple:
#   (date, user_id, video_id, author_id, tab, duration_ms, long_view)
#     x[0]   x[1]     x[2]      x[3]     x[4]    x[5]         x[6]

enc, dim = encode(splits)                   # enc[name] = (X, y, users); dim = total vocab size
X, y, users = enc['train']                  # X: int32 (N,5), y: float32 (N,), users: list
```

To reach any column the kit ignores, edit `data.load()` (add it to the row tuple) and
`data.encode()`'s `raw()` (add it to `FIELDS`), or bypass the kit entirely and hand your own
`scores` array to `evaluate()` — the scoring is fully decoupled from how you build features.
