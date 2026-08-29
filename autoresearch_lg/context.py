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
        "id": "time_and_drift",
        "keywords": ["hourmin", "date", "drift", "time feature", "temporal"],
        "note": "Time features (hourmin, date) and train->test distribution drift.",
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
- Seed averaging on the final submission is cheap variance insurance."""


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

    lines.append(f"Recent iterations (full detail, most recent last):")
    for h in recent:
        err = f" error={h['error'][:200]}" if h.get("error") else ""
        lines.append(
            f"  #{h['iteration']} [{h['mode']}/{h['outcome']}] {h['concept_id']}: "
            f"{h['description']} -> valid={h['metrics']['valid_primary']:.4f} "
            f"test={h['metrics']['test_primary']:.4f}{err}"
        )

    lines.append(f"\nBest valid primary so far: {state['best_valid_primary']:.6f} "
                  f"(checkpoint {state['best_checkpoint_id']})")
    return "\n".join(lines)
