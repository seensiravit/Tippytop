"""Prompt construction. All feedback is valid-only by construction (no test)."""
from __future__ import annotations

from .contract import SOLUTION_CONTRACT

_PROBLEM = """\
KuaiRand-Pure: within-user ranking of short-video impressions. Label = long_view
(0/1). Metric = mean(GAUC, nDCG@5) computed per user on the validation split. Only
the relative order of scores within each user matters. The official Factorization
Machine baseline scores validation primary ~0.60 / test ~0.5946; the oracle ceiling
is ~0.8645. Your job: iteratively improve the pipeline to raise the VALIDATION
primary above the baseline. Promising directions (measured): a ranking loss
(BPR / listwise) instead of pointwise logloss; within-user-varying features
(watch time, engagement signals); per-user sequences. Static user-only features and
bigger embedding dims are known dead-ends.
"""


def system_prompt() -> str:
    return (
        "You are an autonomous ML research engineer improving a recommender "
        "pipeline. Each turn you output a short hypothesis followed by ONE "
        "complete, runnable `solution.py` in a ```python code fence. Obey the "
        "solution contract exactly. Change one idea at a time and explain it in "
        "the hypothesis.\n\n" + SOLUTION_CONTRACT
    )


def data_summary(data) -> str:
    """Compact, read-only description of the dataset (train+valid only)."""
    def bal(split):
        y = data.y(split)
        pos = float(y.sum()); n = len(y)
        return f"{split}: {n:,} rows, {pos/n:.3f} positive" if n else f"{split}: 0"
    return ("FIELDS = ['user_id','video_id','author_id','tab','dur_bucket']; "
            f"embedding-table dim={data.dim}. "
            + bal("train") + "; " + bal("valid") + ".")


def history_digest(records, k: int = 6) -> str:
    """Last k iterations as (iter, valid primary, hypothesis) — valid-only."""
    if not records:
        return "(no prior iterations)"
    lines = []
    for r in records[-k:]:
        vp = r.valid_primary()
        vp_s = "—" if vp is None else f"{vp:.4f}"
        tag = "" if r.error_kind is None else f" [{r.error_kind}]"
        lines.append(f"  iter {r.iter}: valid primary {vp_s}{tag} — {_one(r.hypothesis)}")
    return "\n".join(lines)


def improve_prompt(*, data_sum: str, best_code: str, best_metrics: dict | None,
                   history: str) -> str:
    bp = "—" if not best_metrics else f"{best_metrics.get('primary'):.4f}"
    return (
        f"{_PROBLEM}\nDATA: {data_sum}\n\n"
        f"Current BEST solution (valid primary {bp}):\n"
        f"```python\n{best_code}\n```\n\n"
        f"Recent history (validation only):\n{history}\n\n"
        "Propose ONE improvement over the best solution. Output a hypothesis line, "
        "then the full updated `solution.py` in a ```python fence."
    )


def debug_prompt(*, data_sum: str, failing_code: str, error_kind: str,
                 stderr_tail: str, history: str) -> str:
    return (
        f"{_PROBLEM}\nDATA: {data_sum}\n\n"
        f"The previous solution FAILED ({error_kind}). Fix it. Error output:\n"
        f"```\n{stderr_tail}\n```\n\n"
        f"Failing solution:\n```python\n{failing_code}\n```\n\n"
        f"Recent history (validation only):\n{history}\n\n"
        "Output a one-line hypothesis describing the fix, then a corrected, "
        "complete `solution.py` in a ```python fence."
    )


def _one(s: str, n: int = 70) -> str:
    s = " ".join((s or "").split())
    return s if len(s) <= n else s[: n - 1] + "…"
