"""Invariants for the ranking objectives in ``tippytop.losses.ranking``.

These are pure-numpy and need no dataset, so they run in the fast suite.
The shift-invariance test is the important one: GAUC and nDCG only read the
ORDER of scores within a user, so a correct ranking loss must be blind to
adding a constant to a user's scores. If that breaks, the loss has silently
started spending capacity on absolute score levels the metric never looks at.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tippytop.losses.ranking import (  # noqa: E402
    group_bounds, listwise_softmax_grad, hybrid_grad, bpr_grad,
)


# users: 0 -> 4 impressions (2 pos), 1 -> 3 (1 pos), 2 -> 2 (all negative)
CODES = np.array([0, 1, 0, 2, 1, 0, 0, 1, 2])
Y_RAW = np.array([1, 0, 1, 0, 1, 0, 0, 0, 0], np.float32)
Z_RAW = np.array([0.5, -1.0, 2.0, 0.3, 0.1, 0.0, -0.2, 1.5, 0.7], np.float32)


@pytest.fixture
def grouped():
    order, offs, lens = group_bounds(CODES)
    return order, offs, lens, Y_RAW[order], Z_RAW[order]


def test_group_bounds_partitions_every_row(grouped):
    order, offs, lens, _, _ = grouped
    assert lens.sum() == len(CODES)
    assert sorted(order.tolist()) == list(range(len(CODES)))
    assert np.all(np.diff(CODES[order]) >= 0)          # groups are contiguous


def test_listwise_gradient_sums_to_zero_per_group(grouped):
    """No net push on a user's overall score level — only on its ordering."""
    _, offs, lens, y, z = grouped
    g, _ = listwise_softmax_grad(z, y, offs, lens)
    for o, l in zip(offs, lens):
        assert abs(float(g[o:o + l].sum())) < 1e-6


def test_listwise_is_shift_invariant(grouped):
    """Adding a constant inside a user changes neither loss nor gradient."""
    _, offs, lens, y, z = grouped
    g, loss = listwise_softmax_grad(z, y, offs, lens)
    shifted = z.copy()
    for o, l in zip(offs, lens):
        shifted[o:o + l] += 7.0
    g2, loss2 = listwise_softmax_grad(shifted, y, offs, lens)
    assert np.isclose(loss, loss2)
    assert np.allclose(g, g2)


def test_zero_positive_group_contributes_no_gradient(grouped):
    """Users with no positives have no ordering to learn (metric scores them 0)."""
    _, offs, lens, y, z = grouped
    g, _ = listwise_softmax_grad(z, y, offs, lens)
    o, l = offs[2], lens[2]                            # the all-negative user
    assert float(np.abs(g[o:o + l]).sum()) == 0.0


def test_listwise_prefers_correct_ordering(grouped):
    """Ranking positives above negatives must score a lower loss."""
    _, offs, lens, y, _ = grouped
    good = np.where(y > 0, 5.0, -5.0).astype(np.float32)
    bad = -good
    _, loss_good = listwise_softmax_grad(good, y, offs, lens)
    _, loss_bad = listwise_softmax_grad(bad, y, offs, lens)
    assert loss_good < loss_bad


def test_hybrid_alpha_one_matches_pointwise_direction(grouped):
    """alpha=1 must reduce to the FM baseline objective (up to row scaling)."""
    _, offs, lens, y, z = grouped
    g, _ = hybrid_grad(z, y, offs, lens, alpha=1.0)
    expected = (1.0 / (1.0 + np.exp(-z)) - y) / len(z)
    assert np.allclose(g, expected, atol=1e-6)


def test_hybrid_alpha_zero_matches_listwise(grouped):
    _, offs, lens, y, z = grouped
    g_h, _ = hybrid_grad(z, y, offs, lens, alpha=0.0)
    g_l, _ = listwise_softmax_grad(z, y, offs, lens)
    assert np.allclose(g_h, g_l)


def test_bpr_skips_users_without_a_valid_pair(grouped):
    """BPR's blind spot is exactly GAUC's: all-positive / all-negative users."""
    _, offs, lens, y, z = grouped
    g, _ = bpr_grad(z, y, offs, lens, np.random.default_rng(0))
    o, l = offs[2], lens[2]
    assert float(np.abs(g[o:o + l]).sum()) == 0.0
    assert abs(float(g.sum())) < 1e-6                  # every pair is zero-sum
