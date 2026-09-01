# Devpost draft — Tippytop (TechJam 2026, Track 2)

All fields are filled from `results/final_run/resource_report.json`. Do not
estimate them — copy from the file.

---

## What it does

An autonomous ML research agent for within-user ranking on KuaiRand-Pure. It
reads the task, proposes a hypothesis, writes the code, trains, scores itself on
validation, judges the result against its own noise floor, and decides whether to
tune the idea, expand it, or abandon it — then repeats until it converges or
exhausts its budget. No human picks the next experiment.

## Built with

**Development tools:** VS Code on Windows 11; PowerShell; Git/GitHub; LangGraph
Studio for visualising and stepping the agent graph; pytest for the test suite
(197 tests). No notebooks — every result in this project comes from a script
that can be re-run.

**APIs:** the Anthropic Messages API (`claude-sonnet-5`) for hypothesis
generation and code writing, called through tool-use with a strict JSON schema
so proposals are structured rather than parsed out of prose. The OpenAI
Chat Completions API is supported as an alternative provider (`--model gpt-*`)
but was not used for the submission run.

**Datasets and assets:** **KuaiRand-Pure** only — the six CSVs shipped by the
organisers (Zenodo record 10439422), used exactly as given. Train
2022-04-08→04-21 (1,141,112 rows), validation 04-22→04-28 (124,909), hidden test
04-29→05-08 (170,588); label `long_view`. No external data, no pretrained
weights, no manual labelling. Item- and author-level engagement aggregates are
computed **from the training split only**, leave-one-out per row; we deliberately
did *not* use the shipped `video_features_statistic_pure.csv`, because its
aggregation window is undocumented and may span the test period.

- **Python 3.11**, dependencies managed with **uv**
- **LangGraph** — the agent is a graph, not a `while` loop: `propose → experiment
  → critic → router → check_convergence`, with the experiment stage as its own
  sub-graph so every step has an explicit failure edge
- **Anthropic Claude** (`claude-sonnet-5`) for proposal and code generation
- **NumPy** for the models; **LightGBM** + **scikit-learn** for the tree lane
- **SQLite** for experiment checkpoints, JSONL for the run log

## Results

Primary metric is `mean(GAUC, nDCG@5)`, within-user.

| Model | valid GAUC | valid nDCG@5 | valid primary | test primary | Δ vs official baseline |
|---|---|---|---|---|---|
| random (harness sanity) | 0.4996 | 0.4511 | 0.4753 | 0.4753 | −0.1193 |
| item popularity | 0.6308 | 0.5121 | 0.5715 | 0.5715 | −0.0231 |
| **official FM baseline** | — | — | — | **0.5946** | — |
| FM, our reproduction (10 seeds) | 0.6674 | 0.5363 | 0.6015 ± 0.0006 | 0.5949 ± 0.0008 | +0.0003 |
| FFM, k=4 (6 seeds) | 0.6676 | 0.5362 | 0.6025 ± 0.0004 | 0.5967 ± 0.0004 | **+0.0021** |
| **6×FM + 6×FFM, rank-averaged** | **0.6711** | **0.5380** | **0.6045** | **0.5976** | **+0.0030** |
| oracle ceiling | 1.0000 | 0.7289 | 0.8645 | 0.8645 | +0.2699 |

### Scored per metric

The judging formula is `score = mean over m of (agent(m) - baseline(m))` for
`m ∈ {GAUC, nDCG@5}` on the hidden test set. Our numbers against the official
baseline (`baseline_scores.json`):

| metric (hidden test) | official baseline | ours | **delta** |
|---|---|---|---|
| GAUC | 0.6610 | 0.6646 | **+0.0036** |
| nDCG@5 | 0.5282 | 0.5307 | **+0.0025** |
| primary (mean of the two) | 0.5946 | 0.5976 | **+0.0030** |
| **score_dataset** = mean of the two deltas | | | **+0.00305** |

Because `primary` is *defined* as `mean(GAUC, nDCG@5)`, the mean of the two
per-metric deltas is algebraically identical to the delta in primary — the two
rows agree by construction, not by coincidence. Per-metric values come from
`resource_report.json` (`test_delta_vs_baseline`, `score_dataset`).

**Read the headroom honestly.** The ceiling is 0.8645, not 1.0. 27.1% of test
users are all-negative (nDCG pinned to 0) and 9.2% all-positive (pinned to 1), so
about 36% of users cannot be ranked at all and are excluded from GAUC entirely.
The baseline already captures ~31% of the attainable range; +0.0032 is ~1.2% of
what is left.

**And one caveat we are not hiding.** 6FM+6FFM beats 6FFM alone by +0.0005 on
test, which is inside the evaluation noise (paired-bootstrap SE ≈ 0.0008 over
users). The defensible claim is *FFM plus ensembling beats FM*, established on
validation over disjoint seed halves (+0.0014 / +0.0013 against a 0.0004
threshold). The choice between the two ensembles is not.

## Resource usage

| | |
|---|---|
| Iterations used | **5** of 50 |
| Wall clock | **1084.6 s ≈ 0.30 h** of 6 h budget |
| LLM tokens | **37,379 in / 26,104 out** |
| GPU hours | **0** — CPU-only throughout; FM trains in ~60 s |
| Concepts opened / confirmed | **2 / 1** |

## Manual interventions

**0 interventions.** The number is derived from `interventions.jsonl`, not
declared: `cli run` records a `resume` whenever it starts against a run directory
that already holds iterations, whether or not the operator admits it, and
`cli note "<reason>"` records the ones the harness cannot see. `finalize` copies
the count *and every reason* into `resource_report.json`, and
`scripts/package_final_run.py` refuses to package a run whose reported count
disagrees with its own log.

An earlier version of this repo reported a hardcoded `0`: the field existed on
the state object and reached the report, but nothing ever incremented it. A zero
that is asserted is worth nothing to a judge. A zero that comes from a counter
demonstrably able to go up is evidence.

[If any interventions occurred, list them here with what each one was.]

## How we kept the validation wall honest

The brief says develop on train + validation and keep test for the final pick, so
the interesting question is not whether we intended to — it is which path could
have leaked without anyone noticing.

`test_primary` is computed every iteration, because reporting it is required. It
is written to `runs.jsonl` and never shown to the model: `context.build_context`
formats validation only, by construction.

That left exactly one hole. When a run crashes, the last 2000 characters of its
stdout become `failure_error`, and `build_context` *does* feed that back to the
proposing model — that is how the agent learns what broke. But `baseline.py`
prints its summary block, including the test primary, before some crashes occur.
So a crash at the wrong moment would have put a test metric into the prompt that
chooses the next experiment.

That tail is now scrubbed on its way out (`src/tippytop/runlog/redact.py`), and
`tests/test_run_integrity.py` asserts three things: that the unscrubbed tail
really does carry test signal, that the scrubbed one does not, and that the
validation line survives — over-scrubbing would destroy the signal we rely on.

## Robustness — what happens when things break

Autonomy is not "it has a try/except". It is that every failure has a path that
either recovers or degrades, and none that stops and waits for a person. We
classified the failures by who owns them:

- **Transient provider errors** (429, 529 overloaded, read timeouts) retry with
  exponential backoff, at two levels: the SDK retries the *request*, a LangGraph
  `RetryPolicy` retries the *node*. The SDK cannot recover from a
  malformed-but-successful response; the node-level policy can.
- **Malformed model output** — no tool-use block, a missing key, unparseable
  code — raises a typed error and regenerates, up to 3 attempts (AIDE's
  published `search.max_debug_depth`).
- **Code that crashes** is *repaired*, not rerolled. We classify the traceback
  first: a `NameError` gets the error handed back to the model for a fix
  (MLE-STAR's debugging module), while something that looks stochastic — NaN, a
  singular matrix, a timeout — gets exactly one free reseed before we start
  paying for repairs. Rerolling a seed on a deterministic bug is a guaranteed
  no-op that still costs a full training run.
- **Terminal failure ships anyway.** A provider that stays down routes to
  `finalize`; `run` finalizes in a `finally`; and `cli finalize` can rebuild the
  deliverables from disk at any time. This is the one that mattered: before it,
  a run that died at iteration 30 of 50 produced no `submission.csv` and no
  `resource_report.json`, so thirty completed iterations scored zero on
  Deliverables 3 and 4.

Two bugs surfaced while building this, and both were the kind that look like
correct behaviour from the outside:

1. **Crashes counted toward the plateau.** With `n_plateau=3`, three broken
   experiments in a row converged the run and shipped early — the agent stopping
   because it hit bugs, not because it had run out of ideas.
2. **The budget check was backward-looking.** It asked whether the budget was
   already spent, so at 5h58m it would start an experiment that runs to the
   10-minute cap and leave nothing for finalize.

Every recovery is written to `recovery.jsonl` and summarised in
`resource_report.json` as `recovery_events`, `recovery_by_action` and
`stop_reason`, because a run that recovered silently looks exactly like a run
that never had a problem.

| | |
|---|---|
| Recovery events | **2** |
| Of which repairs / reseeds / retries | **1 repair, 1 finalize-early** |
| Why the run stopped | **plateau** — validation best moved ≤ 0.002 over 3 consecutive iterations |

## What we learned

Most of our value came from *closing* directions with controlled measurements,
not from opening new ones. Three of the organisers' own suggested directions are
closed in `results/leaderboard.md` with the evidence:

**Ranking losses — the objective is better, and it still loses.** Against a
pointwise control at *identical* batching, a listwise loss wins 7 out of 7
configurations, by +0.0008 to +0.0033. But a pairwise or listwise loss needs a
user's impressions inside one batch, and grouped batching costs −0.0023 to
−0.0040 on its own, because it makes that user's embedding updates highly
correlated within a single Adam step. The gain never repays the machinery. Our
first sweep got this backwards for a different reason worth stating: `groups_per_
batch` was hardcoded, so every grouped run trained at a quarter of the baseline's
batch size — the confound, not the loss, was doing the work.

**Multi-task auxiliary heads — 6 of 6 against their own control.** `is_click`
correlates **0.760** with `long_view`, so it is close to a duplicate task: it
consumes embedding capacity and adds almost no information. The score degrades
*monotonically* with auxiliary weight, which is the signature of a wrong
direction rather than a tuning failure. With ~42 impressions per user the
embedding table is the scarce resource.

**Static user features — provably zero, and we can show why.** `user_lv_rate`
scores *exactly* 0.5000 GAUC alone. A feature that is constant across a user's
impressions shifts all their scores equally and cannot reorder anything, and only
within-user order is scored. This also explains why LightGBM LambdaRank
underperforms here (0.5887 valid): the personalised crosses are too sparse to
generalise, so the tree is left with global features and learns one ranking
function for everybody. FM beats it because the signal lives in sparse
user × item interaction — embeddings generalise over that, axis-aligned splits
cannot. Which is also why **FFM**, which models that interaction more finely, was
the direction that actually paid.

## Limitations

- **We are noise-limited, not idea-limited.** Seed-to-seed variation is now
  0.0002–0.0003, but evaluation noise — the CI from resampling the ~24k
  evaluation users — is ≈0.0008. More seeds cannot tighten that. Any future gain
  below ~0.002 is not distinguishable from noise at this sample size, which is
  why we report paired-bootstrap intervals rather than differences of scalars.
- **Validation→test transfer is unmeasured.** We treat a validation gain as
  predictive of a test gain, but we have not estimated the slope, and the
  evidence we do have (FFM: +0.0009 valid, +0.0019 test) is not 1:1.
- **`time_ms` is untouched.** Within-session position varies *within* a user, so
  unlike every user-side aggregate it can reorder — the most obvious remaining
  direction, and one the agent has not yet been given.
