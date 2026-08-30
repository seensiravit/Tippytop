"""FM trained with a ranking objective instead of pointwise logloss.

Same second-order FM as ``fm.py`` — identical forward pass, identical Adam
backward (reused via ``_FMCore.backward``). The ONLY change is what the loss
asks the scores to do:

    fm            each row's score -> its 0/1 label          (pointwise)
    fm_listwise   each user's scores -> softmax over their    (listwise)
                  own impressions, CE against the positives
    fm_bpr        within a user, positive score > negative    (pairwise)

That is the whole hypothesis: GAUC and nDCG@5 only read the *order* of scores
inside a user, so training should optimise order directly rather than spend
capacity forcing absolute values the metric never looks at.

Batching is by USER GROUP, not by row — a listwise/pairwise loss needs a
user's impressions together in one batch to be computable at all.

Users with 0 positives or 0 negatives are dropped from training by default
(``discriminative_only``): they carry no ranking signal, and the metric is
blind to them too (GAUC excludes them; their nDCG@5 is pinned at 0 or 1). On
KuaiRand-Pure that is ~36% of users, so the gradient budget lands entirely on
the population either metric can actually move.
"""
from __future__ import annotations
import time
import numpy as np

from .base import Model
from .fm import _FMCore
from ..data.dataset import Dataset
from . import register
from ..kit import evaluate
from ..losses.ranking import (group_bounds, listwise_softmax_grad, bpr_grad,
                              hybrid_grad)


class _FMRank(Model):
    """Shared trainer; subclasses only pick the objective."""

    name = "fm_rank"
    loss = "listwise"

    def __init__(self, k=16, lr=0.001, l2=1e-6, epochs=40, groups_per_batch=256,
                 patience=4, seed=0, alpha=0.5, list_size=8,
                 discriminative_only=True, verbose=True):
        self.k, self.lr, self.l2 = k, lr, l2
        self.epochs, self.groups_per_batch = epochs, groups_per_batch
        self.patience, self.seed, self.alpha = patience, seed, alpha
        self.list_size = list_size
        self.discriminative_only = discriminative_only
        self.verbose = verbose
        self._m: _FMCore | None = None

    # -- grouping ---------------------------------------------------------

    def _train_groups(self, users, y):
        """Row order + per-group (offset, length), restricted to trainable users."""
        _, codes = np.unique(np.asarray(users), return_inverse=True)
        order, offs, lens = group_bounds(codes)

        n_pos = np.add.reduceat(y[order], offs)
        keep = np.ones(len(offs), dtype=bool)
        if self.discriminative_only:
            keep = (n_pos > 0) & (n_pos < lens)
        if self.verbose:
            print(f"  groups: {len(offs):,} users -> {int(keep.sum()):,} trainable "
                  f"({100 * keep.mean():.1f}%), {lens.mean():.1f} impressions/user")
        return order, offs[keep], lens[keep]

    # -- one optimisation step over a batch of whole user groups ----------

    def _sample_lists(self, order, offs, lens, rng):
        """Draw a short list of impressions per user; return (rows, lengths).

        Two reasons to subsample rather than use a user's whole 43-impression
        history as one list:

        1. Decorrelation. Feeding all of a user's rows through one Adam step
           makes that user's embedding updates highly correlated — measured at
           -0.0023 valid primary against row-shuffled batching, holding the
           objective fixed. Short lists restore stochasticity.
        2. Train/eval list-length match. The eval splits average ~7.1
           impressions per user, so training on ~8-item lists puts the softmax
           over the same list length it is scored on.

        Sampling is uniform within the user, which preserves their natural
        positive rate. Degenerate all-positive / all-negative draws contribute
        no gradient by construction (see ``listwise_softmax_grad``).
        """
        ls = self.list_size
        if not ls:
            return (np.concatenate([order[o:o + l] for o, l in zip(offs, lens)]),
                    lens)
        take = np.minimum(lens, ls)
        rows = np.concatenate([
            order[o:o + l] if l <= ls
            else order[o + rng.choice(l, size=ls, replace=False)]
            for o, l in zip(offs, lens)])
        return rows, take

    def _batch_step(self, m, Xtr, ytr, order, offs, lens, rng):
        rows, lens = self._sample_lists(order, offs, lens, rng)
        b_off = np.r_[0, np.cumsum(lens)[:-1]]
        Xb, yb = Xtr[rows], ytr[rows]

        z, E, S = m.logits(Xb)
        if self.loss == "listwise":
            g, loss = listwise_softmax_grad(z, yb, b_off, lens)
        elif self.loss == "hybrid":
            g, loss = hybrid_grad(z, yb, b_off, lens, alpha=self.alpha)
        elif self.loss == "bpr":
            g, loss = bpr_grad(z, yb, b_off, lens, rng)
        else:                                            # pragma: no cover
            raise ValueError(f"unknown loss {self.loss!r}")

        m.backward(Xb, g, E, S)
        return loss

    # -- Model contract ---------------------------------------------------

    def fit(self, data: Dataset) -> "_FMRank":
        Xtr, ytr, utr = data.enc["train"]
        Xva, yva, uva = data.enc["valid"]
        order, offs, lens = self._train_groups(utr, ytr)

        m = _FMCore(data.dim, k=self.k, lr=self.lr, l2=self.l2, seed=self.seed)
        rng = np.random.default_rng(self.seed)
        best, best_state, bad = -1.0, None, 0

        # Subsampled lists cover only list_size/mean_len of the rows per pass;
        # run enough passes that one "epoch" still sees ~every training row,
        # keeping epoch counts and early stopping comparable to the baseline.
        passes = 1 if not self.list_size else \
            max(1, int(round(lens.mean() / self.list_size)))

        for ep in range(1, self.epochs + 1):
            t0, losses = time.time(), []
            for _ in range(passes):
                perm = rng.permutation(len(offs))
                for i in range(0, len(perm), self.groups_per_batch):
                    sel = perm[i:i + self.groups_per_batch]
                    losses.append(self._batch_step(m, Xtr, ytr, order,
                                                   offs[sel], lens[sel], rng))

            va = evaluate(uva, yva, m.predict(Xva))
            if self.verbose:
                print(f"  epoch {ep:2d} | loss {np.mean(losses):.4f} | "
                      f"valid GAUC {va['GAUC']:.4f} nDCG@5 {va['nDCG@5']:.4f} "
                      f"primary {va['primary']:.4f} | {time.time() - t0:.1f}s")

            if va["primary"] > best + 1e-5:
                best, bad = va["primary"], 0
                best_state = (m.V.copy(), m.W.copy(), np.float32(m.b))
            else:
                bad += 1
                if bad >= self.patience:
                    if self.verbose:
                        print(f"  early stop at epoch {ep}")
                    break

        m.V, m.W, m.b = best_state
        self._m = m
        return self

    def predict(self, data: Dataset, split: str) -> np.ndarray:
        if self._m is None:
            raise RuntimeError("fit() must be called before predict()")
        return self._m.predict(data.enc[split][0])


@register("fm_listwise")
class FMListwise(_FMRank):
    """Softmax over each user's ~7 impressions, CE against their positives.

    Preferred over pairwise here: the eval splits average ~7.1 impressions per
    user, so nDCG@5 covers 5 of ~7 items and the metric is nearly a full-list
    ordering measure — a per-user softmax is both trivial to compute and an
    almost exact match for what is scored.
    """
    name = "fm_listwise"
    loss = "listwise"


@register("fm_bpr")
class FMBpr(_FMRank):
    """Pairwise BPR with negatives sampled inside the user's own impressions."""
    name = "fm_bpr"
    loss = "bpr"


@register("fm_hybrid")
class FMHybrid(_FMRank):
    """``alpha`` * pointwise + (1 - alpha) * listwise. See ``hybrid_grad``."""
    name = "fm_hybrid"
    loss = "hybrid"
