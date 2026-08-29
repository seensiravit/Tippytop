"""Ranking objectives — the #1 headroom direction.

Each returns the per-row gradient wrt the model logits, so any model that
exposes logits can swap objectives without changing its forward pass.

Why listwise first (not pairwise):
    The eval splits average ~7 impressions per user (test: 170,588 rows /
    23,875 users). nDCG@5 therefore covers 5 of ~7 items — the metric is much
    closer to a full-list ordering measure than a top-k-of-many retrieval
    measure. A softmax over a user's ~7 impressions is computationally trivial
    AND corresponds almost exactly to the scored objective, whereas BPR only
    approximates it through sampled pairs. See docs / task.md.

Group representation used throughout: rows are pre-sorted so each user's rows
are contiguous. A group is then (offset, length) into that sorted array —
the layout ``np.add.reduceat`` / ``np.repeat`` consume directly, which keeps
every op vectorised over the whole batch (no Python loop per user).
"""
from __future__ import annotations
import numpy as np


# --------------------------------------------------------------------------
# grouping helper
# --------------------------------------------------------------------------

def group_bounds(codes: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sort row indices by group code; return (order, offsets, lengths).

    ``order``   : row indices such that ``codes[order]`` is non-decreasing.
    ``offsets`` : start of each group *within* ``order``.
    ``lengths`` : size of each group.
    """
    order = np.argsort(codes, kind="stable")
    srt = codes[order]
    starts = np.flatnonzero(np.r_[True, srt[1:] != srt[:-1]])
    lengths = np.diff(np.r_[starts, len(srt)])
    return order, starts, lengths


def _segment_softmax(z: np.ndarray, offsets: np.ndarray,
                     lengths: np.ndarray) -> np.ndarray:
    """Softmax computed independently inside each contiguous group."""
    zc = z - np.repeat(np.maximum.reduceat(z, offsets), lengths)   # stabilise
    e = np.exp(zc)
    return e / np.repeat(np.add.reduceat(e, offsets), lengths)


# --------------------------------------------------------------------------
# objectives
# --------------------------------------------------------------------------

def pointwise_logloss_grad(z: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Baseline objective: sigmoid(z) - y. (Reference — what FM already uses.)"""
    return 1.0 / (1.0 + np.exp(-z)) - y


def listwise_softmax_grad(z: np.ndarray, y: np.ndarray, offsets: np.ndarray,
                          lengths: np.ndarray) -> tuple[np.ndarray, float]:
    """Listwise softmax cross-entropy over each user's own impressions.

    For user u with impressions I_u and positives P_u:

        L_u = -(1/|P_u|) * sum_{i in P_u} log softmax(z)_i
        dL_u/dz_k = p_k - 1[k in P_u] / |P_u|

    The gradient sums to zero within every group, i.e. the loss is invariant to
    adding a constant to a user's scores. That is exactly the invariance GAUC
    and nDCG have, so no capacity is spent learning absolute score levels.

    ``z``, ``y`` are the batch's rows already laid out group-contiguously.
    Returns (grad wrt z, mean loss over groups).
    """
    p = _segment_softmax(z, offsets, lengths)
    n_pos = np.add.reduceat(y, offsets)
    inv = np.where(n_pos > 0, 1.0 / np.maximum(n_pos, 1e-12), 0.0)
    target = y * np.repeat(inv, lengths)          # sums to 1 per group

    # A group with no positives has no ordering to learn: contribute nothing
    # rather than a stray "push everything down" gradient. (Mirrors the metric,
    # which scores those users 0 regardless of prediction.)
    live = np.repeat((n_pos > 0).astype(np.float32), lengths)
    n_groups = max(int((n_pos > 0).sum()), 1)

    grad = ((p - target) * live / n_groups).astype(np.float32)
    loss = -float(np.sum(target * np.log(p + 1e-12))) / n_groups
    return grad, loss


def hybrid_grad(z: np.ndarray, y: np.ndarray, offsets: np.ndarray,
                lengths: np.ndarray, alpha: float = 0.5,
                ) -> tuple[np.ndarray, float]:
    """Convex mix of pointwise logloss and the listwise softmax objective.

        grad = alpha * pointwise + (1 - alpha) * listwise

    Rationale: pure listwise underperforms pointwise on this split. With ~43
    training impressions per user and a high positive rate, the softmax target
    spreads mass over ~17 positives (1/17 each) — a diffuse signal that teaches
    ordering but little else. Pointwise supplies sharp per-row supervision that
    shapes the embeddings; listwise supplies the ordering the metric actually
    reads. Both gradients are mean-normalised (per row / per group) before
    mixing so ``alpha`` is a meaningful dial rather than a scale artefact.

    alpha=1 reduces to the FM baseline objective, alpha=0 to pure listwise.
    """
    n_rows = max(len(z), 1)
    g_point = (pointwise_logloss_grad(z, y) / n_rows).astype(np.float32)
    g_list, loss_list = listwise_softmax_grad(z, y, offsets, lengths)

    grad = (alpha * g_point + (1.0 - alpha) * g_list).astype(np.float32)

    s = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
    loss_point = float(-np.mean(y * np.log(s + 1e-9) + (1 - y) * np.log(1 - s + 1e-9)))
    return grad, alpha * loss_point + (1.0 - alpha) * loss_list


def bpr_grad(z: np.ndarray, y: np.ndarray, offsets: np.ndarray,
             lengths: np.ndarray, rng: np.random.Generator,
             ) -> tuple[np.ndarray, float]:
    """Pairwise BPR restricted to within-user negatives.

    One (positive, negative) pair is sampled per positive, drawn from that same
    user's own impressions — never the global catalogue. The original BPR
    samples negatives from the whole item universe, which is wrong here: we
    only ever rank a user's own logged impressions against each other.

    Kept as the comparison arm for the listwise objective. Note BPR is
    structurally blind to all-positive / all-negative users (no valid pair
    exists) — which is harmless, because GAUC excludes exactly those users and
    their nDCG@5 is pinned at 1 or 0 regardless of prediction.
    """
    pos_idx, neg_idx = [], []
    for off, ln in zip(offsets, lengths):
        yg = y[off:off + ln]
        p_local = np.flatnonzero(yg)
        n_local = np.flatnonzero(yg == 0)
        if len(p_local) == 0 or len(n_local) == 0:
            continue                                    # inert user, no pair
        pos_idx.append(off + p_local)
        neg_idx.append(off + rng.choice(n_local, size=len(p_local)))

    grad = np.zeros_like(z, dtype=np.float32)
    if not pos_idx:
        return grad, 0.0

    pi = np.concatenate(pos_idx)
    ni = np.concatenate(neg_idx)
    d = z[pi] - z[ni]
    sig = 1.0 / (1.0 + np.exp(np.clip(d, -30, 30)))     # = sigmoid(-d)
    n_pairs = len(pi)

    np.add.at(grad, pi, (-sig / n_pairs).astype(np.float32))
    np.add.at(grad, ni, (sig / n_pairs).astype(np.float32))

    loss = -float(np.mean(np.log(1.0 / (1.0 + np.exp(np.clip(-d, -30, 30))) + 1e-12)))
    return grad, loss
