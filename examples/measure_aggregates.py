"""Isolate the item-aggregate direction with a control and an interval.

    python examples/measure_aggregates.py --data_dir KuaiRand-Pure/data

Why this script exists
----------------------
``data/aggregates.py`` is the one direction the organisers' ablation does not
actually cover: `ablation_features.py` opens only `video_features_basic_pure.csv`
and tests four *categorical* IDs, all redundant given `video_id`. Continuous
engagement rates are a different kind of feature and were never tested.

But the aggregates are currently only reachable *inside* `ffm` and `lgbm_rank`,
so their contribution has never been separated from the model that consumes them.
FFM is a verified win at +0.0009 valid / +0.0019 test — and we cannot yet say how
much of that is field-awareness and how much is the aggregate features. Those are
two different claims, and only one of them is currently supported.

This runs the comparison the same way the batching finding was established: an
ablation with everything else held fixed, plus a paired bootstrap over users so
the answer carries an interval rather than being a bare difference of scalars.

What it reports
---------------
1. Whether each aggregate feature actually varies *within* a user. A feature that
   is constant across a user's impressions shifts all their scores equally and
   cannot reorder anything — this is exactly why the organisers' categorical
   ablation measured nothing, and it is worth checking before spending an
   iteration.
2. A screened, seed-confirmed block over the models that use aggregates, against
   an FM control, with both noise sources reported.
"""
from __future__ import annotations

import argparse

import numpy as np

from tippytop.data.dataset import load_dataset
from tippytop.data.aggregates import build_features, FEATURE_NAMES
from tippytop.data.features import varies_within_user
from tippytop.training.runner import train_model
from tippytop.experiments import Candidate, run_block
from tippytop.stats import diagnose_eval_split


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--split", default="valid")
    ap.add_argument("--confirm-seeds", type=int, default=3)
    a = ap.parse_args()

    data = load_dataset(a.data_dir)
    users, labels = data.users(a.split), data.y(a.split)

    print("=" * 68)
    print("1. Can the metric move on this split?")
    d = diagnose_eval_split(users, labels)
    print(f"   {d['users']:,} users | discriminative {100*d['discriminative_frac']:.1f}%"
          f" | can move: {d['metric_can_move']}")
    if not d["metric_can_move"]:
        print("   Degenerate split — stopping, no comparison here is meaningful.")
        return

    print("\n2. Do the aggregate features vary WITHIN a user?")
    print("   (a feature constant inside a user cannot reorder anything, which is")
    print("    why the organisers' categorical ablation measured nothing)")
    feats, names = build_features(data.splits)
    F = feats[a.split]
    usable = []
    for j, nm in enumerate(names):
        rep = varies_within_user(F[:, j], users)
        flag = "usable" if rep["usable"] else "CONSTANT within user"
        print(f"   {nm:22s} varies for {100*rep['fraction_varying']:5.1f}% of users  [{flag}]")
        if rep["usable"]:
            usable.append(nm)
    print(f"   -> {len(usable)}/{len(names)} features can actually affect the ranking.")

    print("\n3. Ablation block: do the aggregates earn their place?")
    print("   FM control vs the models that consume the aggregates.")

    def model(name, **kw):
        def fn(seed):
            return train_model(name, data, seed=seed, verbose=False, **kw).predict(data, a.split)
        return fn

    cands = [Candidate("ffm", model("ffm")), Candidate("lgbm_rank", model("lgbm_rank"))]
    r = run_block(
        "do train-only item aggregates beat plain FM?",
        cands, Candidate("fm", model("fm")),
        users, labels,
        screen_seeds=1, confirm_seeds=a.confirm_seeds, n_boot=2000)

    print("\n" + r.summary())
    print(f"   accepted: {r.accepted}")
    for arm in r.arms:
        if not arm.failed:
            print(f"     {arm.name:12s} seeds={len(arm.seeds)} mean={arm.mean:.4f} sd={arm.std:.4f}")
        else:
            print(f"     {arm.name:12s} FAILED: {(arm.error or '').splitlines()[0]}")

    print("\nNote: this compares model families, so a win here is 'FFM+aggregates beats")
    print("FM', not 'aggregates beat no-aggregates'. To separate the two you need an")
    print("FFM variant with the aggregate columns zeroed as the control — that is the")
    print("same discipline that produced the batching finding.")


if __name__ == "__main__":
    main()
