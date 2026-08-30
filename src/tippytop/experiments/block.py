"""Two-stage experimental block — one iteration that returns a conclusion, not a scalar.

The problem this solves
-----------------------
The budget has two constraints and they are wildly asymmetric: **50 iterations**
(hard cap) against roughly **540 model runs** of compute (6 h at ~40 s each). The
loop currently spends about one train per iteration, so it saturates the binding
constraint while leaving ~90% of the compute idle — and every conclusion it
reaches rests on a single noisy scalar.

Nothing in the rules says an iteration is one model train. An iteration is a step
of the improvement loop; what happens inside it is a design choice. Spending the
abundant resource to buy certainty on the scarce one is the single highest-value
structural change available.

This also fixes premature convergence as a side effect. Runs currently die after
three iterations because three noisy scalars in a row fail to clear
epsilon = 0.002. A block that concludes "listwise beats pointwise, +0.0015,
95% CI [0.0004, 0.0026]" advances the trajectory on evidence rather than on a
coin flip.

Why two-stage rather than uniform
---------------------------------
A uniform 3-seed x 4-config block is 12 runs = 480 s, and 50 of those is 6.7 h —
which *exceeds* the ceiling. Screening cheaply and spending seeds only on
survivors is both cheaper and better targeted:

    stage 1  screen   4 configs x 1 seed   = 160 s
    stage 2  confirm  promote winner,
                      top up to 3 seeds    =  80 s
                                   block   = 240 s  (4 min)

    50 blocks = 3.3 h  ->  56% of the ceiling, 2.7 h margin

Each block returns four screened configurations *and* one powered conclusion.

Design constraint: this **wraps** training rather than replacing it. A candidate
is any callable ``seed -> scores``, so it composes with the existing model
registry, with generated solutions, and with anything added later, without those
paths needing to know this module exists.
"""
from __future__ import annotations

import time
import traceback
from dataclasses import dataclass, field, asdict
from typing import Callable, Sequence

import numpy as np

from ..stats.bootstrap import (PerUserStats, per_user_stats, primary_from,
                               paired_bootstrap, diagnose_eval_split)
from ..stats.power import recommend_seeds, selection_inflation

#: seed -> predicted scores for the evaluation split, in row order.
ScoreFn = Callable[[int], np.ndarray]


@dataclass
class Candidate:
    """One configuration to test. ``fn(seed)`` returns eval-split scores."""
    name: str
    fn: ScoreFn
    meta: dict = field(default_factory=dict)


@dataclass
class ArmResult:
    name: str
    seeds: list[int]
    primaries: list[float]
    mean: float
    std: float
    failed: bool = False
    error: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class BlockResult:
    """The conclusion of one iteration: an effect size with two intervals."""
    hypothesis: str
    arms: list[ArmResult]
    winner: str | None
    baseline_name: str
    delta_vs_baseline: float | None
    bootstrap: dict | None
    seed_std: float | None
    runs_used: int
    wall_s: float
    verdict: str
    accepted: bool

    def to_dict(self) -> dict:
        d = asdict(self)
        d["arms"] = [a.to_dict() for a in self.arms]
        return d

    def summary(self) -> str:
        """One line for a run log — effect size and both noise sources."""
        if self.winner is None:
            return f"[block] {self.hypothesis}: no arm completed. {self.verdict}"
        b = self.bootstrap
        ci = (f" [95% CI {b['ci_low']:+.4f}, {b['ci_high']:+.4f}]" if b else "")
        sd = f" seed-sd {self.seed_std:.4f}" if self.seed_std is not None else ""
        return (f"[block] {self.hypothesis}: {self.winner} "
                f"{self.delta_vs_baseline:+.4f}{ci}{sd} "
                f"({self.runs_used} runs, {self.wall_s:.0f}s) -> {self.verdict}")


def _run_arm(cand: Candidate, seeds: Sequence[int], user_ids, labels,
             cache: dict) -> tuple[ArmResult, dict[int, PerUserStats]]:
    """Train and score one candidate across seeds, tolerating failure.

    A candidate that raises must not kill the block — robustness is graded, and a
    six-hour run cannot afford to die on one bad configuration.
    """
    prim, stats, used = [], {}, []
    for s in seeds:
        key = (cand.name, s)
        try:
            if key in cache:
                st = cache[key]
            else:
                scores = cand.fn(s)
                st = per_user_stats(user_ids, labels, scores)
                cache[key] = st
            stats[s] = st
            prim.append(primary_from(st))
            used.append(s)
        except Exception as e:                       # noqa: BLE001 - deliberate
            return (ArmResult(cand.name, used, prim, float("nan"), float("nan"),
                              failed=True,
                              error=f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=3)}"),
                    stats)
    arr = np.asarray(prim, dtype=float)
    return (ArmResult(cand.name, used, prim, float(arr.mean()),
                      float(arr.std(ddof=1)) if len(arr) > 1 else 0.0), stats)


def run_block(
    hypothesis: str,
    candidates: Sequence[Candidate],
    baseline: Candidate,
    user_ids,
    labels,
    *,
    screen_seeds: int = 1,
    confirm_seeds: int = 3,
    promote: int = 1,
    seed0: int = 0,
    n_boot: int = 2000,
    epsilon: float = 0.002,
    n_candidates_seen: int = 50,
) -> BlockResult:
    """Screen candidates cheaply, confirm the survivor properly, return a conclusion.

    ``n_candidates_seen`` is the number of candidates the *whole run* will have
    screened, used to price selection inflation honestly — the more you look at,
    the more the winner is inflated by chance alone.
    """
    t0 = time.time()
    cache: dict = {}
    runs = 0

    # A split with no discriminative users yields a constant score: every arm
    # ties at exactly 0.0000 and every interval is [0, 0]. Fail loudly rather
    # than returning a confident-looking null result.
    split_diag = diagnose_eval_split(user_ids, labels)
    if not split_diag["metric_can_move"]:
        return BlockResult(hypothesis, [], None, baseline.name, None, None, None,
                           0, time.time() - t0, split_diag["warning"], False)

    screen_s = [seed0 + i for i in range(screen_seeds)]
    confirm_s = [seed0 + i for i in range(confirm_seeds)]

    # baseline is always confirmed at full seed count: everything is measured
    # against it, so it must be the least noisy quantity in the block.
    base_arm, base_stats = _run_arm(baseline, confirm_s, user_ids, labels, cache)
    runs += len(base_arm.seeds)

    # -- stage 1: screen ---------------------------------------------------
    screened: list[ArmResult] = []
    for c in candidates:
        arm, _ = _run_arm(c, screen_s, user_ids, labels, cache)
        runs += len(arm.seeds)
        screened.append(arm)

    alive = [a for a in screened if not a.failed]
    if not alive:
        return BlockResult(hypothesis, [base_arm] + screened, None, baseline.name,
                           None, None, None, runs, time.time() - t0,
                           "every candidate failed to run", False)

    # -- stage 2: confirm the top `promote` --------------------------------
    alive.sort(key=lambda a: a.mean, reverse=True)
    winners = [a.name for a in alive[:promote]]
    by_name = {c.name: c for c in candidates}

    confirmed: list[ArmResult] = []
    win_stats: dict[str, dict] = {}
    for name in winners:
        arm, st = _run_arm(by_name[name], confirm_s, user_ids, labels, cache)
        runs += max(0, len(arm.seeds) - screen_seeds)   # screening seeds are cached
        confirmed.append(arm)
        win_stats[name] = st

    best = max((a for a in confirmed if not a.failed), key=lambda a: a.mean, default=None)
    arms = [base_arm] + [a for a in screened if a.name not in winners] + confirmed

    if best is None:
        return BlockResult(hypothesis, arms, None, baseline.name, None, None, None,
                           runs, time.time() - t0,
                           "promoted candidate failed on confirmation", False)

    delta = best.mean - base_arm.mean

    # Paired bootstrap on the shared seed, so evaluation noise is measured too.
    shared = seed0
    boot = None
    if shared in win_stats[best.name] and shared in base_stats:
        boot = paired_bootstrap(win_stats[best.name][shared], base_stats[shared],
                                n_boot=n_boot, seed=seed0)

    infl = selection_inflation(n_candidates_seen, seeds_per_candidate=confirm_seeds
                               )["expected_inflation"]

    accepted, verdict = _judge(delta, boot, best.std, epsilon, infl)
    return BlockResult(hypothesis, arms, best.name, baseline.name, delta, boot,
                       best.std, runs, time.time() - t0, verdict, accepted)


def _judge(delta: float, boot: dict | None, seed_std: float | None,
           epsilon: float, inflation: float) -> tuple[bool, str]:
    """Accept only when the effect survives both noise sources and selection.

    Three hurdles, deliberately: the effect must be positive, the paired interval
    must exclude zero, and it must exceed what best-of-N selection would have
    manufactured anyway.
    """
    if delta <= 0:
        return False, f"no improvement ({delta:+.4f})"
    if boot is not None and not boot["significant"]:
        return False, (f"{delta:+.4f} but the paired CI spans zero "
                       f"[{boot['ci_low']:+.4f}, {boot['ci_high']:+.4f}] — inside noise")
    if delta < inflation:
        return False, (f"{delta:+.4f} is below the {inflation:.4f} that best-of-N "
                       "selection manufactures on its own — not distinguishable "
                       "from picking a lucky candidate")
    if delta < epsilon:
        return True, (f"{delta:+.4f} is real (CI excludes zero) but below the "
                      f"{epsilon} bar — accept as incumbent, do not call it a win")
    return True, f"{delta:+.4f} clears the {epsilon} bar with the CI excluding zero"


def plan_block(delta_target: float, n_configs: int, run_seconds: float,
               budget_seconds: float, max_iters: int = 50) -> dict:
    """Size a block before running it — does this plan fit the ceiling?

    Third-order effect worth pricing explicitly: a block design that does not fit
    the wall-clock budget will be truncated mid-run, and a truncated run reports a
    worse checkpoint than a smaller block that finished.
    """
    rec = recommend_seeds(delta_target, n_candidates=max_iters * n_configs)
    confirm = rec["seeds"]
    runs = n_configs + confirm          # screen all, top the winner up
    block_s = runs * run_seconds
    total = block_s * max_iters
    return {
        "confirm_seeds": confirm,
        "runs_per_block": runs,
        "block_seconds": block_s,
        "blocks_affordable": int(budget_seconds // block_s) if block_s else 0,
        "total_if_all_iters": total,
        "fits": bool(total <= budget_seconds),
        "utilisation": total / budget_seconds if budget_seconds else 0.0,
        "note": rec["reason"],
    }
