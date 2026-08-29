"""Field-aware Factorization Machine.

FM gives every feature ONE embedding, used for every interaction it takes part
in. So `user_id`'s vector has to simultaneously explain how that user relates to
videos, to authors, to tabs and to duration buckets — one vector, four different
jobs.

FFM gives each feature a **separate embedding per interacting field**:

    FM   score += <v_i,        v_j>
    FFM  score += <v_{i, f(j)}, v_{j, f(i)}>

so `user_id`'s "toward videos" vector is free to differ from its "toward authors"
vector. That is the structure our own diagnostics said was missing: measured on
this data, per-video and per-author aggregates score 0.5807 / 0.5792 alone while
pure user-side features score *exactly* random (GAUC 0.5000) — all the signal is
in user x item interaction, and FFM models that interaction more finely.

FFM won the Criteo, Avazu and Outbrain CTR competitions
(Juan et al., RecSys 2016, https://www.csie.ntu.edu.tw/~cjlin/papers/ffm.pdf).

Parameter count is F x that of FM (F = 5 fields here), so **k must be smaller**:
the paper's guidance is that FFM needs a fraction of FM's dimension, and the
organisers' own ablation showed capacity is not the bottleneck on 1.14M rows.
Default k=4 keeps total parameters close to FM's k=16 while spending them on
structure instead of width.
"""
from __future__ import annotations

import time

import numpy as np

from .base import Model
from ..data.dataset import Dataset
from . import register
from ..kit import evaluate
from ..losses.ranking import group_bounds, listwise_softmax_grad


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


class _FFMCore:
    """V has shape (dim, F, k): one embedding per (feature, interacting field)."""

    def __init__(self, dim, n_fields, k=4, lr=0.001, l2=1e-6, seed=0):
        rng = np.random.default_rng(seed)
        self.F = n_fields
        self.V = rng.normal(0, 0.01, (dim, n_fields, k)).astype(np.float32)
        self.W = np.zeros(dim, dtype=np.float32)
        self.b = np.float32(0.0)
        self.lr, self.l2 = lr, l2
        self.mV = np.zeros_like(self.V); self.vV = np.zeros_like(self.V)
        self.mW = np.zeros_like(self.W); self.vW = np.zeros_like(self.W)
        self.t = 0
        self._pairs = [(a, b) for a in range(n_fields) for b in range(a + 1, n_fields)]

    def logits(self, X):
        E = self.V[X]                                   # (B, F, F, k)
        z = self.b + self.W[X].sum(1)
        for a, b in self._pairs:
            z = z + (E[:, a, b, :] * E[:, b, a, :]).sum(1)
        return z, E

    def backward(self, X, g, E):
        """Adam step from g = dLoss/dlogits."""
        gV = np.zeros_like(self.V)
        gW = np.zeros_like(self.W)
        np.add.at(gW, X, g[:, None])
        gcol = g[:, None].astype(np.float32)
        for a, b in self._pairs:
            # d/dV[X[:,a], b] = E[:, b, a]   and   d/dV[X[:,b], a] = E[:, a, b]
            np.add.at(gV, (X[:, a], b), gcol * E[:, b, a, :])
            np.add.at(gV, (X[:, b], a), gcol * E[:, a, b, :])
        gV += self.l2 * self.V
        gW += self.l2 * self.W

        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for P, G, M, Vv in ((self.V, gV, self.mV, self.vV),
                            (self.W, gW, self.mW, self.vW)):
            M *= b1; M += (1 - b1) * G
            Vv *= b2; Vv += (1 - b2) * (G * G)
            P -= self.lr * (M / (1 - b1 ** self.t)) / (np.sqrt(Vv / (1 - b2 ** self.t)) + eps)
        self.b -= self.lr * g.sum()

    def predict(self, X, bs=100_000):
        return np.concatenate([self.logits(X[i:i + bs])[0]
                               for i in range(0, len(X), bs)])


@register("ffm")
class FFM(Model):
    """Pointwise-trained FFM. Directly comparable to `fm`."""

    name = "ffm"
    loss = "pointwise"

    def __init__(self, k=4, lr=0.001, l2=1e-6, epochs=40, bs=8192, patience=4,
                 groups_per_batch=256, list_size=32, seed=0, verbose=True, **kw):
        self.k, self.lr, self.l2 = k, lr, l2
        self.epochs, self.bs, self.patience = epochs, bs, patience
        self.groups_per_batch, self.list_size = groups_per_batch, list_size
        self.seed, self.verbose = seed, verbose
        self._m: _FFMCore | None = None

    # -- pointwise: plain shuffled row batches, same regime as `fm` ---------

    def _epoch_pointwise(self, m, Xtr, ytr, rng):
        idx = rng.permutation(len(ytr))
        losses = []
        for i in range(0, len(idx), self.bs):
            sel = idx[i:i + self.bs]
            Xb, yb = Xtr[sel], ytr[sel]
            z, E = m.logits(Xb)
            p = _sigmoid(z)
            m.backward(Xb, ((p - yb) / len(yb)).astype(np.float32), E)
            losses.append(float(-np.mean(yb * np.log(p + 1e-9)
                                         + (1 - yb) * np.log(1 - p + 1e-9))))
        return float(np.mean(losses))

    # -- listwise: whole-user lists, batch size held at bs -------------------

    def _epoch_listwise(self, m, Xtr, ytr, order, offs, lens, rng):
        # groups_per_batch is derived so rows/batch stays ~self.bs regardless of
        # list_size -- the confound that made earlier group-batched runs train
        # at a quarter of the baseline's batch size.
        gpb = max(1, self.bs // max(self.list_size, 1))
        perm = rng.permutation(len(offs))
        losses = []
        for i in range(0, len(perm), gpb):
            sel = perm[i:i + gpb]
            o, l = offs[sel], np.minimum(lens[sel], self.list_size)
            rows = np.concatenate([
                order[s:s + n] if n >= ln else
                order[s + rng.choice(ln, size=n, replace=False)]
                for s, n, ln in zip(o, l, lens[sel])])
            b_off = np.r_[0, np.cumsum(l)[:-1]]
            Xb, yb = Xtr[rows], ytr[rows]
            z, E = m.logits(Xb)
            g, loss = listwise_softmax_grad(z, yb, b_off, l)
            m.backward(Xb, g, E)
            losses.append(loss)
        return float(np.mean(losses))

    def fit(self, data: Dataset) -> "FFM":
        Xtr, ytr, utr = data.enc["train"]
        Xva, yva, uva = data.enc["valid"]
        n_fields = Xtr.shape[1]
        m = _FFMCore(data.dim, n_fields, k=self.k, lr=self.lr, l2=self.l2,
                     seed=self.seed)
        rng = np.random.default_rng(self.seed)

        grouped = self.loss == "listwise"
        if grouped:
            _, codes = np.unique(np.asarray(utr), return_inverse=True)
            order, offs, lens = group_bounds(codes)
            keep = (np.add.reduceat(ytr[order], offs) > 0)
            offs, lens = offs[keep], lens[keep]

        if self.verbose:
            print(f"  FFM k={self.k} x {n_fields} fields = "
                  f"{m.V.size:,} interaction params ({self.loss})")

        best, best_state, bad = -1.0, None, 0
        for ep in range(1, self.epochs + 1):
            t0 = time.time()
            loss = (self._epoch_listwise(m, Xtr, ytr, order, offs, lens, rng)
                    if grouped else self._epoch_pointwise(m, Xtr, ytr, rng))
            va = evaluate(uva, yva, m.predict(Xva))
            if self.verbose:
                print(f"  epoch {ep:2d} | loss {loss:.4f} | valid GAUC "
                      f"{va['GAUC']:.4f} nDCG@5 {va['nDCG@5']:.4f} "
                      f"primary {va['primary']:.4f} | {time.time()-t0:.1f}s")
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


@register("ffm_listwise")
class FFMListwise(FFM):
    """FFM trained with the within-user softmax objective.

    Combines the two things measured separately on this branch: the ranking loss
    is worth +0.0010..+0.0017 against a matched control, and field-aware
    interactions model the user x item signal that carries everything here.
    Batch size is held at `bs` rows regardless of `list_size`.
    """
    name = "ffm_listwise"
    loss = "listwise"
