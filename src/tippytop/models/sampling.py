"""Within-user pair sampling and minibatch helpers."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Sequence

import numpy as np


def build_pair_indices(
    labels: np.ndarray,
    users: Sequence[str],
    *,
    pairs_per_positive: int = 1,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample positive-negative row pairs from the same user."""
    grouped: dict[str, tuple[list[int], list[int]]] = defaultdict(lambda: ([], []))
    for index, (user, label) in enumerate(zip(users, labels, strict=True)):
        grouped[user][0 if label > 0.5 else 1].append(index)

    rng = np.random.default_rng(seed)
    positive_parts: list[np.ndarray] = []
    negative_parts: list[np.ndarray] = []
    for positives, negatives in grouped.values():
        if not positives or not negatives:
            continue
        # Ranking metrics compare impressions within a user, never across users.
        repeated = np.repeat(np.asarray(positives, dtype=np.int64), pairs_per_positive)
        sampled = rng.choice(np.asarray(negatives, dtype=np.int64), size=len(repeated), replace=True)
        positive_parts.append(repeated)
        negative_parts.append(sampled)
    if not positive_parts:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    return np.concatenate(positive_parts), np.concatenate(negative_parts)


def iter_batches(indices: np.ndarray, batch_size: int) -> Iterable[np.ndarray]:
    for start in range(0, len(indices), batch_size):
        yield indices[start : start + batch_size]
