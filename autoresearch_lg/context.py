"""build_context() and retrieve_options() — the two functions technical-plan.md
calls the actual payoff of the propose sub-graph: keeping up to 50 iterations
of history legible to the LLM, and surfacing the right next direction without
hardcoding a curriculum (that would script the agent and kill the Autonomy
score — the LLM always decides *what*, these just narrow *where to look*).
"""
from __future__ import annotations

# README.md's own ranked "从哪里开始改" list (§6 of technical-plan.md), kept
# as data so retrieve_options can filter it — NOT a schedule. The agent is
# free to ignore this ordering or go beyond it entirely once explored.
HEADROOM = [
    {
        "id": "loss_function",
        "keywords": ["listwise", "softmax", "pairwise", "bpr", "ranking loss", "rank-aligned"],
        "note": (
            "Loss-function alignment: pointwise log-loss -> listwise softmax "
            "within-user, or pairwise BPR (negatives restricted to each "
            "user's own impressions). Organizers' highest-priority direction "
            "— GAUC/nDCG are rank metrics, current loss isn't."
        ),
    },
    {
        "id": "user_sequences",
        "keywords": ["sequence", "attention", "din", "history", "pooling"],
        "note": (
            "User-history sequences: attention-weighted pooling over recent "
            "interactions (DIN-style; do NOT softmax-normalize the weights). "
            "~42 events/user on average — temper expectations vs. papers "
            "assuming hundreds."
        ),
    },
    {
        "id": "multi_task",
        "keywords": ["multi-task", "multitask", "auxiliary", "click", "like", "follow", "comment", "forward"],
        "note": (
            "Multi-task: shared-embedding auxiliary heads on is_click/"
            "is_like/is_follow/is_comment/is_forward alongside long_view "
            "(all fully observed here — not an ESMM-style causal chain)."
        ),
    },
    {
        "id": "censored_watch_time",
        "keywords": ["watch time", "censored", "play_time_ms", "duration regression"],
        "note": (
            "Censored watch-time regression on play_time_ms with a one-sided "
            "loss (playback gets truncated at video end). Sits inside the "
            "multi-task step; higher research depth, higher risk."
        ),
    },
    {
        "id": "architecture",
        "keywords": ["deepfm", "dcn", "xdeepfm", "architecture", "capacity"],
        "note": (
            "Model architecture (DeepFM/DCN/xDeepFM). LAST resort — the "
            "project's own ablation shows capacity/architecture is not the "
            "bottleneck (larger k barely moved the score)."
        ),
    },
    {
        "id": "field_aware",
        "keywords": ["ffm", "field-aware", "field aware", "per-field embedding"],
        "note": (
            "Field-aware FM (FFM): give each feature a SEPARATE embedding per "
            "interacting field rather than one embedding for all interactions, "
            "so user_id's 'toward videos' vector can differ from its 'toward "
            "authors' vector. Use a smaller k (4 vs FM's 16) so total parameters "
            "stay comparable -- spend them on structure, not width. Trains "
            "row-shuffled, so it pays none of the grouped-batching cost a "
            "listwise loss does. VERIFIED here over 6 seeds each: FFM valid "
            "0.6025 +-0.0004 / test 0.5967 +-0.0004 vs FM valid 0.6016 +-0.0003 "
            "/ test 0.5948 +-0.0009 -- non-overlapping, every FFM seed beat "
            "every FM seed on test, and FFM's test variance is less than half. "
            "This is the strongest single model measured. Won "
            "Criteo/Avazu/Outbrain (Juan et al., RecSys 2016)."
        ),
    },
    {
        "id": "ensemble_diversity",
        "keywords": ["ensemble", "blend", "rank-average", "seed averaging", "stacking"],
        "note": (
            "Rank-average several models WITHIN-USER (not raw scores -- scales "
            "differ between objectives and only order is scored). Verified: "
            "+0.0013..+0.0014 valid on two disjoint seed halves, on top of "
            "whatever the members score. Note it adds roughly the same amount "
            "for same-family and cross-family members (6x FFM 0.6032 vs "
            "3FM+3FFM 0.6035), so pick STRONG members first; mixing families "
            "is a marginal extra, not the main effect."
        ),
    },
    {
        "id": "gbdt_lambdarank",
        "keywords": ["lightgbm", "lambdarank", "gbdt", "boosting", "lambdamart"],
        "note": (
            "LightGBM with objective='lambdarank': query groups map exactly "
            "onto users, and it optimises NDCG directly by weighting each pair "
            "by the metric change a swap would cause. ~7 impressions/user is "
            "squarely its regime. Needs real per-row features to split on, so "
            "it pairs naturally with train-only item aggregates. Untried."
        ),
    },
    {
        "id": "item_aggregates",
        "keywords": ["aggregate", "target encoding", "item statistic", "play_progress",
                     "count feature", "smoothing"],
        "note": (
            "Train-only per-video aggregates: long_view rate, mean play "
            "progress, impression count, smoothed toward the global mean for "
            "rare videos. These are ITEM-side, so they vary within a user's "
            "list and CAN move a within-user ranking. Note the 'static features "
            "yield nothing' result is narrower than it reads: "
            "ablation_features.py only opens video_features_basic_pure.csv and "
            "only tests 4 categorical IDs, which are redundant given video_id. "
            "Continuous engagement rates were never tested. Compute these from "
            "the TRAIN split yourself rather than reading "
            "video_features_statistic_pure.csv, whose aggregation window is "
            "unknown and may span the test period."
        ),
    },
    {
        "id": "time_and_drift",
        "keywords": ["hourmin", "date", "drift", "time feature", "temporal", "position"],
        "note": (
            "Time features and drift. time_ms is unused and gives WITHIN-SESSION "
            "POSITION when sorted per user-day — position bias is one of the "
            "largest effects in any feed and varies within a user, so it can "
            "move the ranking. hourmin crossed with duration is a second angle."
        ),
    },
    {
        "id": "unbiased_eval",
        "keywords": ["log_random", "unbiased", "randomized exposure"],
        "note": (
            "log_random_4_22_to_5_08_pure.csv is a randomized-exposure log — "
            "free bias-free validation signal, checks whether a change only "
            "wins on biased traffic. Least-attempted direction."
        ),
    },
]

DATASET_EDGES = """\
Dataset-derived edges (judge-checkable, not in any shared README):
- 36.3% of users are inert: 27.1% all-negative (nDCG pinned to 0, unsalvageable),
  9.2% all-positive (nDCG pinned to 1). Consider upweighting movable users in
  the TRAINING loss only — never touch how eval scores them, that's fixed.
- Eval lists are short (~7 impressions/user) — listwise objectives are cheap here.
- Seed averaging on the final submission is cheap variance insurance.

Measured, and it changes how you must READ a ranking-loss result:
- A listwise/pairwise loss needs a user's rows in one batch. That BATCHING alone
  costs about -0.0023 valid primary, measured with the objective held fixed at
  pointwise. The ranking objective itself is worth +0.0010..+0.0017 (4 of 4
  configs). So a CORRECT ranking loss compared naively against the FM baseline
  looks like a failure. If you try one, also run a pointwise control at the SAME
  batching and compare against that, not against the baseline.
  (Cause: Adam normalises per-parameter, so a user embedding gets one step per
  epoch when grouped instead of ~43 when rows are shuffled.)
- Single-model spread over 10 seeds: valid 0.6015 +- 0.0006, test 0.5949 +-
  0.0008. Seed 42 alone gives 0.6019/0.5957 — a full sigma high. Over many
  iterations, best-of-N on that noise manufactures apparent gains, so re-run
  anything that clears epsilon at 2-3 seeds before believing it.
- Rank-averaging several models gives a replicated +0.0013 (two disjoint seed
  groups, monotonic in ensemble size). Real, but below the 0.002 bar — and it
  does not compound across members of ONE family, which share a bias. Six
  FM+FFM members beat twelve FM-only members: diversity of family is what pays.

Directions already CLOSED here with controlled measurements — do not re-propose
these without a genuinely new angle, and say what the angle is:
- Ranking loss (listwise/BPR) on FM: the objective wins 7 of 7 matched
  comparisons, but grouped batching costs more than it returns. Best matched
  listwise 0.5987 vs baseline 0.6019. Closed.
- Multi-task auxiliary heads: 6 of 6 weight settings fail to beat their own
  w=0 control, degrading monotonically as auxiliary weight rises. is_click
  correlates 0.760 with long_view — a near-duplicate task that consumes
  embedding capacity for no new information. Five other signals (is_follow,
  is_comment, is_forward, is_hate, profile_stay_time) are 0.0-0.3% nonzero and
  carry no usable gradient at all. Closed.
- GBDT/LambdaRank over engineered aggregates: valid 0.5887. Per-feature scores
  say why — video_lv_rate alone reaches 0.5807, but user_lv_rate scores EXACTLY
  random GAUC 0.5000 (user-side features are constant within a user, so they
  cannot reorder anything) and user x author crosses are near-random because the
  pairs are too sparse. The tree only ever sees global features. The signal is
  sparse user x item interaction: embeddings generalise over it, axis-aligned
  splits cannot. Closed as a standalone model; still viable as a diverse
  ensemble member or via stacking on an FM score.

Still untried: user history sequences (~42 events/user, temper expectations),
and unbiased evaluation against log_random_4_22_to_5_08_pure.csv (1.18M
randomly-exposed rows — one extra evaluation pass, no new modelling)."""


def _closed_concept_text(state) -> str:
    return " ".join(
        (c["statement"] + " " + c.get("closed_reason", "")).lower()
        for c in state["concepts"] if c["status"] == "closed"
    )


def retrieve_options(state) -> str:
    """Filter HEADROOM to what hasn't obviously been tried/closed already,
    formatted as options — never a forced next step."""
    tried_text = _closed_concept_text(state)
    remaining = [
        h for h in HEADROOM
        if not any(kw in tried_text for kw in h["keywords"])
    ]
    if not remaining:
        lines = ["All headroom directions below have some closed-concept overlap "
                  "already — reread them anyway, a closed concept may have only "
                  "tried one angle of a broader direction:"]
        remaining = HEADROOM
    else:
        lines = ["Available directions (not a schedule — pick what your reasoning "
                  "and the EDA support, or go beyond this list):"]
    for h in remaining:
        lines.append(f"- {h['note']}")
    lines.append("")
    lines.append(DATASET_EDGES)
    return "\n".join(lines)


def build_context(state) -> str:
    """Summarize run history so the LLM can reason over up to 50 iterations
    without the prompt growing linearly. Recent iterations stay in full
    detail; older ones collapse to per-concept aggregates."""
    history = state["history"]
    if not history:
        return "No experiments yet — this is the first proposal after the baseline run."

    RECENT_N = 8
    recent = history[-RECENT_N:]
    older = history[:-RECENT_N]

    lines = []
    if older:
        by_concept: dict[str, list[dict]] = {}
        for h in older:
            by_concept.setdefault(h["concept_id"], []).append(h)
        lines.append(f"Older iterations ({len(older)}), summarized by concept:")
        for cid, attempts in by_concept.items():
            concept = next((c for c in state["concepts"] if c["id"] == cid), None)
            statement = concept["statement"] if concept else "(unknown)"
            best = max((a["metrics"]["valid_primary"] for a in attempts), default=0.0)
            outcomes = [a["outcome"] for a in attempts]
            status = concept["status"] if concept else "?"
            reason = f" ({concept['closed_reason']})" if concept and concept.get("closed_reason") else ""
            lines.append(
                f"  {cid} [{status}{reason}]: {statement}\n"
                f"    {len(attempts)} attempt(s), outcomes={outcomes}, best valid={best:.4f}"
            )
        lines.append("")

    # Validation only. test_primary is deliberately NOT shown: CONSTRAINTS tells
    # the model never to let test decide what to try next, and the brief is
    # explicit that the agent develops on train + validation alone. It still goes
    # into runs.jsonl via critic.write_log — reporting it is required, feeding it
    # back into the proposal prompt is not.
    lines.append(f"Recent iterations (full detail, most recent last):")
    for h in recent:
        err = f" error={h['error'][:200]}" if h.get("error") else ""
        lines.append(
            f"  #{h['iteration']} [{h['mode']}/{h['outcome']}] {h['concept_id']}: "
            f"{h['description']} -> valid={h['metrics']['valid_primary']:.4f}{err}"
        )

    lines.append(f"\nBest valid primary so far: {state['best_valid_primary']:.6f} "
                  f"(checkpoint {state['best_checkpoint_id']})")
    return "\n".join(lines)
