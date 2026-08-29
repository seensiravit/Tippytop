"""Ensembles: average several models' rankings instead of trusting one.

Two cheap, well-motivated wins the single-model runs leave on the table.

`fm_seedavg` — the same FM at several seeds. The baseline's seed std is 0.0008
    on every metric, so any single run is a draw from a distribution; averaging
    cancels that noise. A run is ~40s, so five seeds cost ~3 minutes.

`fm_blend` — pointwise FM + listwise FM. They optimise different objectives and
    therefore make different mistakes; averaging their rankings keeps what they
    agree on and discards uncorrelated error.

Both combine **within-user ranks, not raw scores**. Score scales are arbitrary
and differ wildly between a logloss-trained and a softmax-trained model, so
averaging raw scores would just let the wider-scaled model dominate. Ranks are
scale-free, and since only within-user order is scored, ranking is the only
information that matters anyway.
"""
from __future__ import annotations
import numpy as np

from .base import Model
from ..data.dataset import Dataset
from . import register, build
from ..losses.ranking import group_bounds


def _rank_within_user(scores: np.ndarray, users) -> np.ndarray:
    """Map scores to their within-user rank, normalised to [0, 1].

    Vectorised: sort by (user, score) once, then read positions back out.
    """
    _, codes = np.unique(np.asarray(users), return_inverse=True)
    order, offs, lens = group_bounds(codes)

    out = np.empty(len(scores), dtype=np.float64)
    for o, l in zip(offs, lens):
        idx = order[o:o + l]
        r = np.empty(l, dtype=np.float64)
        r[np.argsort(scores[idx], kind="stable")] = np.arange(l)
        out[idx] = r / max(l - 1, 1)
    return out


class _Ensemble(Model):
    """Fit several members, then average their within-user ranks."""

    name = "ensemble"
    members: tuple = ()          # (model_name, kwargs) pairs

    def __init__(self, seed=0, verbose=True, **kw):
        self.seed, self.verbose, self.kw = seed, verbose, kw
        self._fitted: list = []

    def _spec(self) -> list[tuple[str, dict]]:
        return list(self.members)

    def fit(self, data: Dataset) -> "_Ensemble":
        self._fitted = []
        for i, (name, kw) in enumerate(self._spec()):
            if self.verbose:
                print(f"  [{i + 1}/{len(self._spec())}] fitting {name} {kw}")
            m = build(name, verbose=False, **kw)
            m.fit(data)
            self._fitted.append(m)
        return self

    def predict(self, data: Dataset, split: str) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("fit() must be called before predict()")
        users = data.users(split)
        ranks = [_rank_within_user(m.predict(data, split), users)
                 for m in self._fitted]
        return np.mean(ranks, axis=0)


@register("fm_seedavg")
class FMSeedAverage(_Ensemble):
    """The baseline FM at N seeds, rank-averaged. Pure variance reduction."""

    name = "fm_seedavg"

    def __init__(self, n_seeds=5, seed=0, k=16, lr=0.001, epochs=40,
                 verbose=True, **kw):
        super().__init__(seed=seed, verbose=verbose, **kw)
        self.n_seeds, self.k, self.lr, self.epochs = n_seeds, k, lr, epochs

    def _spec(self):
        return [("fm", dict(seed=self.seed + i, k=self.k, lr=self.lr,
                            epochs=self.epochs))
                for i in range(self.n_seeds)]


@register("fm_ffm")
class FMPlusFFM(_Ensemble):
    """FM + FFM, rank-averaged across seeds — the first CROSS-FAMILY ensemble.

    Every earlier ensemble here averaged members of one family, which cancels
    seed noise but cannot cancel a bias all members share; measured, they all
    landed within noise of each other (0.6022-0.6030) and did not compound.

    FM and FFM differ structurally, not just by seed: FM gives each feature one
    embedding for every interaction it takes part in, FFM gives it a separate
    embedding per interacting field. They therefore misrank different pairs.
    Verified over 6 seeds each (see results/leaderboard.md):

        fm    valid 0.6016 +- 0.0003   test 0.5948 +- 0.0009
        ffm   valid 0.6025 +- 0.0004   test 0.5967 +- 0.0004

    FFM is the stronger member outright — non-overlapping, every FFM seed beat
    every FM seed on test. Ensembling adds ~+0.0010 on top, replicated on two
    disjoint seed halves. Cross-family over same-family is only +0.0003 at equal
    member count, which is inside the noise: **most of the gain is FFM being a
    better model, not the mixing.** n_seeds=6 (12 members) is the measured best
    on test at 0.5978.
    """

    name = "fm_ffm"

    def __init__(self, n_seeds=6, seed=0, verbose=True, **kw):
        super().__init__(seed=seed, verbose=verbose, **kw)
        self.n_seeds = n_seeds

    def _spec(self):
        spec = []
        for i in range(self.n_seeds):
            spec.append(("fm", dict(seed=self.seed + i, epochs=40)))
            spec.append(("ffm", dict(seed=self.seed + i, epochs=40, k=4)))
        return spec


@register("fm_diverse")
class FMDiverse(_Ensemble):
    """Diverse ensemble: several capacities x both objectives.

    Ensembles gain from members that make *different* errors, not from more
    draws of the same model. The organisers measured k=8/16/32 as individually
    equivalent (0.5895/0.5902/0.5887) — which makes them ideal ensemble members:
    same skill, decorrelated mistakes. Pairing each capacity with both the
    pointwise and the listwise objective adds a second axis of disagreement.
    """

    name = "fm_diverse"

    def __init__(self, seed=0, n_seeds=2, ks=(8, 16, 32), verbose=True, **kw):
        super().__init__(seed=seed, verbose=verbose, **kw)
        self.n_seeds, self.ks = n_seeds, tuple(ks)

    def _spec(self):
        spec = []
        for i in range(self.n_seeds):
            for k in self.ks:
                spec.append(("fm", dict(seed=self.seed + i, k=k, epochs=40)))
                spec.append(("fm_listwise", dict(seed=self.seed + i, k=k,
                                                 epochs=40, lr=0.001,
                                                 groups_per_batch=256,
                                                 list_size=0)))
        return spec


@register("fm_blend")
class FMBlend(_Ensemble):
    """Pointwise FM + listwise FM, rank-averaged across seeds.

    The two objectives disagree in uncorrelated ways; the blend keeps the
    agreement. Uses several seeds of each so it also absorbs the seed variance.
    """

    name = "fm_blend"

    def __init__(self, n_seeds=3, seed=0, verbose=True, **kw):
        super().__init__(seed=seed, verbose=verbose, **kw)
        self.n_seeds = n_seeds

    def _spec(self):
        spec = []
        for i in range(self.n_seeds):
            spec.append(("fm", dict(seed=self.seed + i, epochs=40)))
            spec.append(("fm_listwise", dict(seed=self.seed + i, epochs=40,
                                             lr=0.001, groups_per_batch=256,
                                             list_size=0)))
        return spec
