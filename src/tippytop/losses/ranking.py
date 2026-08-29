"""Ranking objectives (STUBS — the highest-value work stream).

Each returns the per-row gradient wrt the model logits, so any model that
exposes logits can swap objectives without changing its forward pass.
"""
from __future__ import annotations
import numpy as np


def pointwise_logloss_grad(z: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Baseline objective: sigmoid(z) - y. (Reference — already what FM uses.)"""
    return 1.0 / (1.0 + np.exp(-z)) - y


def bpr_grad(*args, **kwargs):
    """Pairwise BPR: sample (user, pos, neg) triplets within a user, maximise
    sigma(z_pos - z_neg). TODO."""
    raise NotImplementedError


def listwise_softmax_grad(*args, **kwargs):
    """Listwise: softmax over each user's impressions, cross-entropy against the
    positives. TODO."""
    raise NotImplementedError
