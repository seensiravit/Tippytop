# How the Baseline Works

`baseline.py` ships **three** scoring models. Only one is a real model (FM); the other two exist as
reference points. All three produce a `score` per eval row and hand it to `evaluate()`.

```bash
python3 baseline.py --model fm       # the model to beat
python3 baseline.py --model pop      # trivial baseline
python3 baseline.py --model random   # lower bound / sanity check
```

| Model | Trains? | Test primary | Purpose |
|---|---|---|---|
| `random` | no | 0.4753 | lower bound — proves the eval harness works |
| `pop` | no (pure counting) | 0.5715 | trivial baseline — "just recommend popular videos" |
| **`fm`** | **yes** | **0.5946** | **the official baseline you must beat** |

Remember the task is **within-user ranking**: for each user you only order *their own* impressions by
`long_view` (0/1). Only the *relative* order of scores within a user matters — never the absolute value,
and never comparisons across users.

---

## 1. `random` — the lower bound

```python
rng.random(len(rws))     # a uniform random score per row
```

Assigns each row a random score. Ranks users' impressions in random order. It exists purely as a
**self-check**: if your eval code doesn't score `random` at primary ≈ 0.475, the harness is broken —
fix that before trusting any real result.

---

## 2. `pop` — item popularity (trivial baseline)

No training, no per-user modeling — just **"how often is this video a long_view, globally?"**

For every video it counts, over the **train** split, how many times it was shown (`imp`) and how many
of those were long-views (`pos`), then scores with a **smoothed positive rate**:

```python
score(v) = (pos[v] + prior * gmean) / (imp[v] + prior)     # prior = 20, gmean = global long_view rate
```

- The `prior * gmean` term is **Bayesian smoothing**: a rarely-seen video is pulled toward the global
  average instead of trusting a noisy 1-out-of-2 rate. `prior = 20` sets how many "pseudo-impressions"
  of average behavior to blend in.
- Every user gets the **same** score for the same video, so within a user this just orders videos by
  global popularity. Surprisingly strong (0.5715) — popularity is a hard-to-beat signal.

---

## 3. `fm` — Factorization Machine (the real baseline)

A **second-order Factorization Machine** trained from scratch in pure numpy with Adam. This is the one
to improve on. Three parts: features → the FM formula → training.

### 3.1 Features (from `data.py`)

Each impression is reduced to **5 categorical fields**:

```
[user_id, video_id, author_id, tab, dur_bucket]
```

`data.encode()` maps every field's values to integer ids (train vocabulary + one `UNK` slot per field)
and lays them out in **one shared embedding table** via per-field offsets. So each row becomes 5 integer
indices into that table.

### 3.2 The scoring formula

Each of the 5 active ids pulls:
- a **scalar weight** `Wᵢ` (first-order / linear term), and
- a length-`k` **embedding vector** `Eᵢ = V[idᵢ]` (`k = 16`).

The score for a row is:

```
score = b  +  Σᵢ Wᵢ  +  ½ · [ (Σᵢ Eᵢ)²  −  Σᵢ Eᵢ² ]
        │      │             └──────── pairwise interactions ────────┘
        │      └ linear term (per-field bias)
        └ global bias
```

The bracketed term is the **FM trick**. The thing we actually want is the sum of dot-products between
every *pair* of the 5 embeddings:

```
Σ_{i<j} ⟨Eᵢ, Eⱼ⟩   =   ½ · [ (Σ Eᵢ)² − Σ Eᵢ² ]
```

The identity on the right computes all pairwise interactions in **linear time** (sum-then-square)
instead of the naive quadratic loop. In code (`FM.logits`):

```python
E = self.V[X]                                    # (B, 5, k)  the 5 embeddings per row
S = E.sum(1)                                      # (B, k)     their sum
inter = 0.5 * ((S ** 2).sum(1) - (E ** 2).sum((1, 2)))
return self.b + self.W[X].sum(1) + inter
```

**Why this matters:** the `user_id × video_id` pairwise term is what lets the model learn
*"this specific user likes this specific video"* — per the README, that cross carries most of the
learnable signal. The other pairs (user×author, video×tab, …) add smaller refinements.

### 3.3 Training

| Setting | Value | Note |
|---|---|---|
| Loss | pointwise **logistic loss** (binary cross-entropy on `sigmoid(score)`) | predicts P(long_view = 1) per row |
| Optimizer | **Adam**, hand-rolled | lr = 0.001, β = (0.9, 0.999) |
| Regularization | L2 = 1e-6 on `V` and `W` | |
| Batch size | 8192 | data reshuffled every epoch |
| Max epochs | 40 | |
| Early stopping | patience = 4 on **validation primary score** | keeps the best snapshot |
| Runtime | ~40 s | CPU, single core |

The loop (`run_fm`):

1. Shuffle the training rows, iterate in batches of 8192; each batch calls `m.step(X, y)`.
2. **`step()`** computes the logits, the gradient `g = sigmoid(z) − y`, scatters that gradient onto only
   the embeddings each row touched (`np.add.at`, sparse update), adds L2, and applies one Adam update to
   `V`, `W`, and `b`. It returns the batch log-loss for logging.
3. After each epoch, score the **valid** split with `evaluate()`. If the primary score improved, save a
   snapshot `(V, W, b)`; otherwise increment a "bad epochs" counter.
4. If validation hasn't improved for **4 epochs**, stop early, restore the best snapshot, and report
   final **valid** and **test** scores.

### 3.4 A subtlety: pointwise loss vs. a ranking metric

The FM is trained with **pointwise** logloss (each row judged independently), but it's *evaluated* with
**ranking** metrics (GAUC / nDCG). That mismatch is exactly the headroom the README flags first: swapping
the loss for a **pairwise (BPR)** or **listwise (per-user softmax)** objective aligns training with how
you're scored — the most promising single change, and it needs no new features.

---

## 4. How predictions become a score

For every model, the flow is identical:

```python
scores = model_predicts(rows)                       # one real number per eval row
evaluate([r.user_id for r in rows],                 # group rows by user
         [r.long_view for r in rows],               # 0/1 labels
         scores)                                     # → {'GAUC', 'nDCG@5', 'primary'}
```

`evaluate()` groups rows by `user_id`, sorts each user's rows by score, and computes GAUC + nDCG@5.
**The model is fully decoupled from scoring** — you can replace FM with anything (PyTorch, LightGBM, a
sequence model) and, as long as you produce a `scores` array in the same row order, `evaluate()` scores
it the same way. See `submit.py` for turning `scores` into a submission CSV.

---

## 5. TL;DR

- **`random`** = noise floor (sanity check the harness).
- **`pop`** = "recommend globally popular videos," no training, smoothed count.
- **`fm`** = a 2nd-order Factorization Machine over 5 categorical fields, trained with Adam + pointwise
  logloss, whose `user_id × video_id` interaction does the heavy lifting. **This 0.5946 is the number to beat.**
- The bottleneck is **not** features or model capacity (both measured, both flat) — it's the
  **objective** (pointwise vs. ranking) and the **unused signals** (behavior sequences, watch time,
  other engagement labels). Start there.
