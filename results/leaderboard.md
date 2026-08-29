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
| 2026-08-29 | auto | fm_listwise (seed=42) | 0.6591 | 0.5319 | 0.5955 | 0.5892 |  |
