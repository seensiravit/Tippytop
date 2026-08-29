"""Factorization Machine baseline — the 0.5946 row to beat.

Ported from the vendored ``baseline.py`` (FM class + run_fm loop) behind the
``Model`` interface. Second-order FM in pure numpy: Adam, L2, pointwise logloss,
early stopping on valid primary with best-snapshot restore. Logic is kept
identical to the kit; this is an adapter, not a new model.
"""
from __future__ import annotations
import time
import numpy as np

from .base import Model
from ..data.dataset import Dataset
from . import register
from ..kit import evaluate


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


class _FMCore:
    """The numpy FM (weights + one Adam step), mirroring baseline.FM."""

    def __init__(self, dim, k=16, lr=0.001, l2=1e-6, seed=0):
        rng = np.random.default_rng(seed)
        self.V = rng.normal(0, 0.01, (dim, k)).astype(np.float32)
        self.W = np.zeros(dim, dtype=np.float32)
        self.b = np.float32(0.0)
        self.lr, self.l2 = lr, l2
        self.mV = np.zeros_like(self.V); self.vV = np.zeros_like(self.V)
        self.mW = np.zeros_like(self.W); self.vW = np.zeros_like(self.W)
        self.t = 0

    def logits(self, X):
        E = self.V[X]                                    # (B, F, k)
        S = E.sum(1)                                     # (B, k)
        inter = 0.5 * ((S ** 2).sum(1) - (E ** 2).sum((1, 2)))
        return self.b + self.W[X].sum(1) + inter, E, S

    def backward(self, X, g, E, S):
        """One Adam update from ``g`` = dLoss/dlogits, and the forward's E, S.

        Split out of ``step`` so any objective (see ``tippytop.losses.ranking``)
        can drive the same FM backward pass. Math is unchanged.
        """
        gV = np.zeros_like(self.V); gW = np.zeros_like(self.W)
        np.add.at(gW, X, g[:, None])
        np.add.at(gV, X, g[:, None, None] * (S[:, None, :] - E))
        gV += self.l2 * self.V; gW += self.l2 * self.W
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for P, G, M, Vv in ((self.V, gV, self.mV, self.vV), (self.W, gW, self.mW, self.vW)):
            M *= b1; M += (1 - b1) * G
            Vv *= b2; Vv += (1 - b2) * (G * G)
            P -= self.lr * (M / (1 - b1 ** self.t)) / (np.sqrt(Vv / (1 - b2 ** self.t)) + eps)
        self.b -= self.lr * g.sum()

    def step(self, X, y):
        B = len(y)
        z, E, S = self.logits(X)
        g = ((_sigmoid(z) - y) / B).astype(np.float32)   # (B,)
        self.backward(X, g, E, S)
        return float(-np.mean(y * np.log(_sigmoid(z) + 1e-9)
                              + (1 - y) * np.log(1 - _sigmoid(z) + 1e-9)))

    def predict(self, X, bs=200_000):
        return np.concatenate([self.logits(X[i:i + bs])[0]
                               for i in range(0, len(X), bs)])


@register("fm")
class FMBaseline(Model):
    name = "fm"

    def __init__(self, k=16, lr=0.001, l2=1e-6, epochs=40, bs=8192,
                 patience=4, seed=0, verbose=True):
        self.k, self.lr, self.l2 = k, lr, l2
        self.epochs, self.bs, self.patience = epochs, bs, patience
        self.seed, self.verbose = seed, verbose
        self._m: _FMCore | None = None

    def fit(self, data: Dataset) -> "FMBaseline":
        Xtr, ytr, _ = data.enc["train"]
        Xva, yva, uva = data.enc["valid"]
        m = _FMCore(data.dim, k=self.k, lr=self.lr, l2=self.l2, seed=self.seed)
        rng = np.random.default_rng(self.seed)
        best, best_state, bad = -1.0, None, 0
        for ep in range(1, self.epochs + 1):
            idx = rng.permutation(len(ytr)); t0 = time.time()
            losses = [m.step(Xtr[idx[i:i + self.bs]], ytr[idx[i:i + self.bs]])
                      for i in range(0, len(idx), self.bs)]
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
