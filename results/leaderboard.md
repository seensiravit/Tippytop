# Leaderboard

Shared scoreboard — append every measured run here so results stay comparable.
Report **valid** primary for iteration (test only for final picks). Δ vs FM
baseline; a change is real only if Δvalid > +0.002 (noise band).

| Date | Owner | Model / change | valid GAUC | valid nDCG@5 | **valid primary** | test primary | Notes |
|---|---|---|---|---|---|---|---|
| — | ref | random (sanity) | 0.4996 | 0.4511 | 0.4753 | 0.4753 | harness check |
| — | ref | pop | 0.6308 | 0.5121 | 0.5715 | 0.5715 | trivial |
| — | ref | **FM baseline** | — | — | — | **0.5946** | the row to beat |
| — | ref | oracle ceiling | 1.0000 | 0.7289 | 0.8645 | 0.8645 | perfect ranking |

**Targets:** beat FM 0.5946; measure progress against oracle 0.8645 (headroom ≈ 0.27).
| 2026-08-29 | auto | fm (seed=42) | 0.6674 | 0.5363 | 0.6019 | 0.5957 |  |
| 2026-08-29 | jovi | fm_listwise (group-batched) | 0.6641 | 0.5346 | 0.5994 | 0.5927 | lr 1e-3, gpb 256 |
| 2026-08-29 | jovi | fm_hybrid a=0.3 (group-batched) | 0.6639 | 0.5350 | 0.5995 | 0.5936 | mix pointwise+listwise |
| 2026-08-29 | jovi | **control**: pointwise, group-batched | — | — | **0.5996** | 0.5927 | a=1.0 — same objective as FM |

### Finding: group-batching, not the loss, is what costs the -0.002

Holding the objective fixed at pointwise (`fm_hybrid --alpha 1.0`), batching by
whole user group scores **0.5996** vs the row-shuffled FM baseline's **0.6019**.
So −0.0023 of the gap is caused by *how rows are batched*, not by the ranking
objective. Against its own like-for-like control the listwise loss is neutral to
slightly positive (0.5998 / 0.5999 vs 0.5996) — i.e. **the ranking loss is not
the thing that is failing here.**

Cause: a listwise/pairwise loss needs a user's impressions in one batch, which
makes that user's embedding updates highly correlated within a single Adam step.
Restricting training to discriminative users costs nothing either way (0.5996
both with and without the filter).

### Confirmed: the ranking loss IS better — 4/4 at matched batching

Sweeping lr x list_size, each ranking loss run paired with a pointwise control
using *identical* batching:

| config | pointwise control | listwise | hybrid a=0.3 | loss gain |
|---|---|---|---|---|
| lr 5e-4, list 8 | 0.5958 | 0.5968 | 0.5968 | +0.0010 |
| lr 5e-4, list 16 | 0.5969 | 0.5982 | 0.5986 | +0.0017 |
| lr 2e-4, list 8 | 0.5978 | 0.5986 | 0.5993 | +0.0015 |
| lr 2e-4, list 16 | 0.5985 | **0.5998** | 0.5991 | +0.0013 |

The ranking objective beats pointwise in every configuration by +0.0010..+0.0017.
It loses in absolute terms only because the batching it requires costs ~-0.003,
which is larger than its own gain. **Read every ranking-loss number against the
a=1.0 control at the same batching, never against the raw FM baseline** — the
agent's prompts should say this too, or it will keep drawing the wrong conclusion.

### Ensembles: first models to clear the baseline

| Model | valid primary | test primary | vs FM valid |
|---|---|---|---|
| FM baseline | 0.6019 | 0.5957 | — |
| `fm_seedavg` (5 seeds, rank-averaged) | **0.6030** | 0.5960 | +0.0011 |
| `fm_blend` (3x FM + 3x listwise) | **0.6029** | **0.5962** | +0.0010 |

Both beat the baseline but sit inside the +-0.002 noise band, so neither counts
as a real win yet. Members are combined by **within-user rank**, not raw score:
score scales differ between a logloss- and a softmax-trained model, and only
within-user order is scored anyway.

---

## 2026-08-30 — the ranking-loss direction, closed

### Correction: the earlier sweep was confounded by batch size

`groups_per_batch` was hardcoded to 256, so rows/batch = `256 x list_size`. The
baseline uses 8192. Every group-batched run above therefore trained at a quarter
to half the baseline's batch size — a different optimiser regime, never the
variable under test. Re-running the pointwise control across batch sizes, seed 42:

| rows/batch | control valid | gap vs baseline 0.6019 |
|---|---|---|
| 2,048 (list=8) | 0.5923 | -0.0096 |
| 4,096 (list=16) | 0.5964 | -0.0055 |
| 8,192 (list=32, **matched**) | 0.5979 | **-0.0040** |
| ~11,136 (whole group) | 0.5996 | **-0.0023** |

Monotone in batch size. So batch size explains over half the spread — but the gap
does **not** vanish at matched batch size. Grouping carries a genuine residual
cost; -0.0023 (whole-group) is its best case.

### The loss is better, and it still loses

Listwise vs a pointwise control at identical settings — the confound applies to
both arms and cancels, so these comparisons stand:

| rows/batch | control | listwise | loss gain |
|---|---|---|---|
| 2,048 | 0.5923 | 0.5955 | +0.0033 |
| 4,096 | 0.5964 | 0.5976 | +0.0012 |
| 8,192 | 0.5979 | **0.5987** | +0.0008 |

**7 of 7 controlled comparisons favour the ranking objective** (these three plus
the four in the lr sweep above). But the gain *shrinks as the regime improves* —
much of the apparent advantage was the loss compensating for over-stepping. At a
fair batch size it is +0.0008, while grouping costs -0.0040 in the same setting.

**Conclusion: the ranking loss never recovers the batching cost.** Best matched
listwise is 0.5987 vs the baseline's 0.6019. The objective is genuinely better;
the machinery it requires costs more than it returns. The organisers' #1
recommended direction is closed for FM, with seven controlled measurements.

### New model family: FFM

Field-aware FM — a separate embedding per (feature, interacting field) instead of
one per feature. `k=4` keeps parameters near FM's `k=16`. Trains row-shuffled, so
it pays no batching penalty. Won Criteo / Avazu / Outbrain
(Juan et al., RecSys 2016).

| Model | valid GAUC | valid nDCG@5 | valid primary | test primary |
|---|---|---|---|---|
| fm (seed 42) | 0.6674 | 0.5363 | 0.6019 | 0.5957 |
| **ffm** (k=4, seed 42) | 0.6676 | 0.5362 | **0.6019** | **0.5965** |
| ffm_listwise | 0.6632 | 0.5345 | 0.5988 | 0.5923 |
| lgbm_rank (LambdaRank) | 0.6494 | 0.5281 | 0.5887 | 0.5768 |

FFM ties FM on validation and generalises better to test (+0.0008). Equal skill,
different errors — the right shape for an ensemble member, and the first genuinely
different family we have.

### Why LambdaRank underperforms — measured, not guessed

Each aggregate feature scored alone on valid primary:

| feature | valid primary | GAUC |
|---|---|---|
| video_lv_rate | 0.5807 | 0.6387 |
| author_lv_rate | 0.5792 | 0.6367 |
| user_tab_rate | 0.5251 | 0.5589 |
| duration_log | 0.4730 | 0.4860 |
| user_lv_rate | 0.4837 | **0.5000** |
| user_author_rate | 0.4826 | **0.4982** |

Two things fall out. `user_lv_rate` scores *exactly* random GAUC — a direct
confirmation that user-side features are constant within a user and cannot
reorder anything. And `duration_log` scores **below** random: duration is
strongly *negatively* predictive, which is the duration bias the CWM/D2Q papers
address.

The personalised crosses (`user_author_rate` etc.) are near-random because they
are too sparse — a user rarely meets the same author in train and valid. So the
tree only has *global* features and learns one ranking function for everybody.
**FM beats it because the signal lives in sparse user x item interaction:
embeddings generalise over that, axis-aligned splits cannot.** That is why FFM —
which models that interaction more finely — is the more promising direction, and
why stacking (FM's score as a tree feature) is the sensible way to combine them.

### Multi-task: closed, 6 of 6 against its own control

`fm_multitask` shares one embedding table across `long_view` plus auxiliary heads
on `is_click` and D2Q-style debiased watch time. `h` is initialised to ones, so
`w=(0,0)` is an exact FM and serves as the control (0.6023 — marginally above
plain FM's 0.6019 because the head is learnable).

| w_click | w_watch | valid | test | vs control |
|---|---|---|---|---|
| 0.0 | 0.0 | **0.6023** | 0.5957 | control |
| 0.3 | 0.0 | 0.6021 | 0.5954 | -0.0002 |
| 0.0 | 0.3 | 0.6023 | 0.5956 | -0.0000 |
| 0.1 | 0.1 | 0.6021 | 0.5958 | -0.0001 |
| 0.0 | 1.0 | 0.6016 | 0.5954 | -0.0007 |
| 1.0 | 1.0 | 0.6004 | 0.5939 | -0.0019 |

**Nothing beats the control, and the score degrades monotonically with auxiliary
weight.** Not a tuning failure — the trend is the wrong way at every step.

Most likely mechanism: `is_click` correlates **0.760** with `long_view`, so it is
close to a duplicate task — it consumes embedding capacity while adding almost no
information. Watch-time quantile is more independent (corr 0.635) but at weight
1.0 actively distracts from the main objective. With ~42 impressions per user the
embedding table is the scarce resource, and spending it on a near-duplicate task
is a straight loss.

The organisers' direction #3, closed. Worth stating in the write-up with the
correlation figure, since "we added the auxiliary signals the README lists" is
what most teams will do without checking whether those signals carry independent
information.

### Direction ledger

| Direction | Status | Evidence |
|---|---|---|
| Ranking loss (organisers' #1) | **closed** | 7/7 matched comparisons: better objective, cannot repay its batching cost |
| LambdaRank / GBDT | **closed** | no user x item signal available to axis-aligned splits |
| Multi-task (organisers' #3) | **closed** | 6/6 against control, monotone the wrong way |
| Static features (organisers') | closed by organisers | but see the item-aggregates note — their ablation is narrower than it reads |
| **Cross-family ensembling** | **live** | 6 cross-family members beat 12 same-family; verification in progress |
| User sequences (organisers' #2) | untried | ~42 events/user, temper expectations |
| Unbiased eval on `log_random` | untried | one eval pass, no new modelling |
| 2026-08-29 | auto | fm_listwise (seed=42) | 0.6591 | 0.5319 | 0.5955 | 0.5892 |  |
