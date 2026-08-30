"""Unified command-line interface for the whole pipeline.

    python -m tippytop run    --model fm
    python -m tippytop submit --model fm --split test --out results/submissions/fm_test.csv
    python -m tippytop check  results/submissions/fm_test.csv --split test
    python -m tippytop score  results/submissions/fm_test.csv --split valid

`run` trains + evaluates on valid & test. `submit` trains then writes a
submission CSV for a split (retrain-on-submit) and validates it. `check`/`score`
validate an existing CSV against the eval set (score is valid-only, since test
labels are for the leaderboard).
"""
from __future__ import annotations
import argparse

from . import config
from .kit import evaluate
from .data.dataset import load_dataset
from .training import train_model, evaluate_model, append_leaderboard
from .submission import write_submission, read_submission


def _model_kwargs(a) -> dict:
    kw = {}
    for key in ("k", "lr", "epochs", "alpha", "list_size", "groups_per_batch"):
        v = getattr(a, key, None)
        if v is not None:
            kw[key] = v
    return kw


def cmd_run(a) -> int:
    data = load_dataset(a.data_dir or config.DATA_DIR)
    print("loaded: " + ", ".join(f"{k}={len(v):,d}" for k, v in data.splits.items()))
    model = train_model(a.model, data, seed=a.seed, **_model_kwargs(a))
    res = evaluate_model(model, data)
    print(f"\n=== {a.model} (seed={a.seed}) ===")
    for sp in ("valid", "test"):
        m = res[sp]
        print(f"  {sp:5s}  GAUC {m['GAUC']:.4f} | nDCG@5 {m['nDCG@5']:.4f} | "
              f"primary {m['primary']:.4f}")
    tp = res["test"]["primary"]
    print(f"\n  test primary {tp:.4f}  vs  FM {config.FM_BASELINE_PRIMARY} | "
          f"oracle {config.ORACLE_CEILING_PRIMARY} "
          f"({100 * (tp - config.FM_BASELINE_PRIMARY):+.2f} pts vs FM)")
    if not a.no_log:
        append_leaderboard(a.model, a.seed, res, note=a.note or "")
    return 0


def cmd_submit(a) -> int:
    data = load_dataset(a.data_dir or config.DATA_DIR)
    print("loaded: " + ", ".join(f"{k}={len(v):,d}" for k, v in data.splits.items()))
    model = train_model(a.model, data, seed=a.seed, **_model_kwargs(a))
    scores = model.predict(data, a.split)
    out = write_submission(a.out, data.splits[a.split], scores)
    print(f"wrote {out}: {len(scores):,d} rows (split={a.split}, model={a.model})")
    read_submission(out, data.splits[a.split])   # self-validate immediately
    print("check passed: format and alignment OK")
    if a.split == "valid":
        rows = data.splits["valid"]
        m = evaluate([x[1] for x in rows], [x[6] for x in rows], scores)
        print(f"  valid  GAUC {m['GAUC']:.4f} | nDCG@5 {m['nDCG@5']:.4f} | "
              f"primary {m['primary']:.4f}")
    return 0


def cmd_check(a) -> int:
    data = load_dataset(a.data_dir or config.DATA_DIR)
    rows = data.splits[a.split]
    read_submission(a.path, rows)
    print(f"check passed: {len(rows):,d} rows, split={a.split}")
    return 0


def cmd_score(a) -> int:
    data = load_dataset(a.data_dir or config.DATA_DIR)
    rows = data.splits[a.split]
    scores = read_submission(a.path, rows)
    print(f"check passed: {len(rows):,d} rows, split={a.split}")
    m = evaluate([x[1] for x in rows], [x[6] for x in rows], scores)
    print(f"  GAUC {m['GAUC']:.4f} | nDCG@5 {m['nDCG@5']:.4f} | "
          f"primary {m['primary']:.4f}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="tippytop",
                                 description="KuaiRand within-user ranking pipeline")
    ap.add_argument("--data_dir", default=None,
                    help="override data dir (default: vendored KuaiRand-Pure/data)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add_model_opts(p):
        p.add_argument("--model", default="fm", help="fm | pop | random | <registered>")
        p.add_argument("--seed", type=int, default=config.DEFAULT_SEED)
        p.add_argument("--k", type=int, default=None, help="FM embedding dim")
        p.add_argument("--lr", type=float, default=None, help="FM learning rate")
        p.add_argument("--epochs", type=int, default=None, help="FM max epochs")
        # ranking models (fm_listwise / fm_bpr / fm_hybrid) — see models/fm_rank.py
        p.add_argument("--alpha", type=float, default=None,
                       help="fm_hybrid: pointwise weight (1.0 = FM objective)")
        p.add_argument("--list_size", type=int, default=None,
                       help="impressions sampled per user per step (0 = whole history)")
        p.add_argument("--groups_per_batch", type=int, default=None,
                       help="users per optimisation step")

    p = sub.add_parser("run", help="train + evaluate on valid & test")
    add_model_opts(p)
    p.add_argument("--no-log", action="store_true", help="do not append to leaderboard")
    p.add_argument("--note", default="", help="leaderboard note")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("submit", help="train + write a submission CSV for a split")
    add_model_opts(p)
    p.add_argument("--split", default="test", choices=["valid", "test"])
    p.add_argument("--out", default=str(config.SUBMISSIONS_DIR / "submission.csv"))
    p.set_defaults(func=cmd_submit)

    p = sub.add_parser("check", help="validate an existing submission CSV")
    p.add_argument("path")
    p.add_argument("--split", default="test", choices=["valid", "test"])
    p.set_defaults(func=cmd_check)

    p = sub.add_parser("score", help="validate + score an existing CSV (valid only)")
    p.add_argument("path")
    p.add_argument("--split", default="valid", choices=["valid", "test"])
    p.set_defaults(func=cmd_score)

    from .agent.cli import register_agent_subparser
    register_agent_subparser(sub)
    return ap


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    return a.func(a)
