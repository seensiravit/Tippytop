"""Multi-task FM — shared embeddings, auxiliary supervision from unused signals.

Every other model here trains on one bit per impression: ``long_view``. The log
carries eleven more feedback signals that the frozen loader drops. With ~42
training impressions per user, the **embedding table is the scarce resource**,
and auxiliary supervision is regularisation that costs no new data — each extra
signal gives the same embeddings more gradient to learn from.

Measured density on train (the reason only two auxiliary heads are used):

    is_click            46.3% nonzero, corr 0.760 with long_view  -> used
    play_time_ms        86.1% nonzero, corr 0.635                 -> used
    is_like              1.9%, is_profile_enter 2.5%              -> marginal
    is_follow/comment/forward/hate   0.0-0.3% nonzero             -> dropped

A signal that is near-constant carries no usable gradient. Adding all of them
because the README lists them would spend capacity on noise.

**This is deliberately not ESMM.** That architecture exists to correct sample
selection bias, where a post-click label is only observed for clicked items. No
such bias exists here: every signal is logged on every impression. The mechanism
being tested is plain shared-bottom regularisation.

Architecture — a strict generalisation of FM:

    E          = V[X]                                  (B, F, k)
    S          = E.sum(1)                              (B, k)
    inter_vec  = 0.5 * (S**2 - (E**2).sum(1))          (B, k)   per-dimension
    z_t        = b_t + W_t[X].sum(1) + inter_vec @ h_t

Ordinary FM is the case ``h_t = ones(k)``, which is how ``h`` is initialised, so
training starts from exactly the baseline and the auxiliary heads can only add.
``V`` is shared across tasks; ``W_t``, ``b_t`` and ``h_t`` are per-task, so each
task reads the shared representation through its own lens.
"""
from __future__ import annotations

import time

import numpy as np

from .base import Model
from ..data.dataset import Dataset
from ..data.signals import load_aux, assert_aligned, watch_quantile
from . import register
from ..kit import evaluate
from .. import config


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


class _MultiTaskFM:
    def __init__(self, dim, k, n_tasks, lr, l2, seed):
        rng = np.random.default_rng(seed)
        self.V = rng.normal(0, 0.01, (dim, k)).astype(np.float32)
        self.W = np.zeros((n_tasks, dim), dtype=np.float32)
        self.b = np.zeros(n_tasks, dtype=np.float32)
        # ones -> task 0 starts as an exact FM; heads can only differentiate.
        self.h = np.ones((n_tasks, k), dtype=np.float32)
        self.lr, self.l2, self.T = lr, l2, n_tasks
        self.mV = np.zeros_like(self.V); self.vV = np.zeros_like(self.V)
        self.mW = np.zeros_like(self.W); self.vW = np.zeros_like(self.W)
        self.mh = np.zeros_like(self.h); self.vh = np.zeros_like(self.h)
        self.t = 0

    def forward(self, X):
        E = self.V[X]                                   # (B, F, k)
        S = E.sum(1)                                    # (B, k)
        inter = 0.5 * (S ** 2 - (E ** 2).sum(1))        # (B, k)
        Z = (self.b[:, None] + self.W[:, X].sum(2)      # (T, B)
             + self.h @ inter.T)
        return Z, E, S, inter

    def step(self, X, targets, weights):
        """One Adam step. ``targets`` (T, B) in [0,1]; ``weights`` (T,)."""
        B = X.shape[0]
        Z, E, S, inter = self.forward(X)
        P = _sigmoid(Z)
        G = ((P - targets) * weights[:, None] / B).astype(np.float32)  # (T, B)

        gV = np.zeros_like(self.V)
        gW = np.zeros_like(self.W)
        # Shared embeddings receive every task's gradient, routed through h.
        Gk = (G.T @ self.h).astype(np.float32)          # (B, k)
        np.add.at(gV, X, (Gk[:, None, :] * (S[:, None, :] - E)))
        for t in range(self.T):
            np.add.at(gW[t], X, G[t][:, None])
        gh = (G @ inter).astype(np.float32)             # (T, k)

        gV += self.l2 * self.V
        gW += self.l2 * self.W

        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for P_, Gr, M, Vv in ((self.V, gV, self.mV, self.vV),
                              (self.W, gW, self.mW, self.vW),
                              (self.h, gh, self.mh, self.vh)):
            M *= b1; M += (1 - b1) * Gr
            Vv *= b2; Vv += (1 - b2) * (Gr * Gr)
            P_ -= self.lr * (M / (1 - b1 ** self.t)) / (np.sqrt(Vv / (1 - b2 ** self.t)) + eps)
        self.b -= self.lr * G.sum(1)

        main = targets[0]
        p0 = P[0]
        return float(-np.mean(main * np.log(p0 + 1e-9)
                              + (1 - main) * np.log(1 - p0 + 1e-9)))

    def predict(self, X, bs=200_000):
        """Task 0 (long_view) only — the auxiliary heads exist to shape V."""
        out = []
        for i in range(0, len(X), bs):
            Xb = X[i:i + bs]
            E = self.V[Xb]
            S = E.sum(1)
            inter = 0.5 * (S ** 2 - (E ** 2).sum(1))
            out.append(self.b[0] + self.W[0][Xb].sum(1) + inter @ self.h[0])
        return np.concatenate(out)


@register("fm_multitask")
class FMMultiTask(Model):
    """FM with auxiliary heads on ``is_click`` and debiased watch time."""

    name = "fm_multitask"

    def __init__(self, k=16, lr=0.001, l2=1e-6, epochs=40, bs=8192, patience=4,
                 w_click=0.3, w_watch=0.3, seed=0, verbose=True, **kw):
        self.k, self.lr, self.l2 = k, lr, l2
        self.epochs, self.bs, self.patience = epochs, bs, patience
        self.w_click, self.w_watch = w_click, w_watch
        self.seed, self.verbose = seed, verbose
        self._m: _MultiTaskFM | None = None

    def _targets(self, data: Dataset) -> np.ndarray:
        aux = load_aux(config.DATA_DIR)
        assert_aligned(aux, data.splits)          # fail loudly, never silently
        tr = aux["train"]
        dur = np.asarray([r[5] for r in data.splits["train"]], dtype=np.float64)
        # D2Q-style: watch time ranked WITHIN its duration bucket, so the target
        # is "longer than others stay on videos this length", not raw seconds.
        wq = watch_quantile(tr, dur)
        return np.stack([data.y("train"), tr["is_click"], wq]).astype(np.float32)

    def fit(self, data: Dataset) -> "FMMultiTask":
        Xtr = data.X("train")
        Xva, yva, uva = data.enc["valid"]
        targets = self._targets(data)
        weights = np.array([1.0, self.w_click, self.w_watch], dtype=np.float32)

        m = _MultiTaskFM(data.dim, self.k, len(weights), self.lr, self.l2, self.seed)
        rng = np.random.default_rng(self.seed)
        if self.verbose:
            print(f"  tasks: long_view (1.0) | is_click ({self.w_click}) | "
                  f"watch_quantile ({self.w_watch})")

        best, best_state, bad = -1.0, None, 0
        for ep in range(1, self.epochs + 1):
            idx = rng.permutation(Xtr.shape[0]); t0 = time.time()
            losses = [m.step(Xtr[idx[i:i + self.bs]], targets[:, idx[i:i + self.bs]],
                             weights)
                      for i in range(0, len(idx), self.bs)]
            va = evaluate(uva, yva, m.predict(Xva))
            if self.verbose:
                print(f"  epoch {ep:2d} | loss {np.mean(losses):.4f} | valid GAUC "
                      f"{va['GAUC']:.4f} nDCG@5 {va['nDCG@5']:.4f} "
                      f"primary {va['primary']:.4f} | {time.time()-t0:.1f}s")
            if va["primary"] > best + 1e-5:
                best, bad = va["primary"], 0
                best_state = (m.V.copy(), m.W.copy(), m.b.copy(), m.h.copy())
            else:
                bad += 1
                if bad >= self.patience:
                    if self.verbose:
                        print(f"  early stop at epoch {ep}")
                    break

        m.V, m.W, m.b, m.h = best_state
        self._m = m
        return self

    def predict(self, data: Dataset, split: str) -> np.ndarray:
        if self._m is None:
            raise RuntimeError("fit() must be called before predict()")
        return self._m.predict(data.enc[split][0])
