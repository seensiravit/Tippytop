"""The shared experiment loop: load -> encode -> fit -> score every split.

Every model goes through here so results are comparable and logged identically.
"""
from __future__ import annotations
from datetime import date

from .. import config
from ..kit import evaluate
from ..data.dataset import load_dataset, Dataset
from ..models import build


def _fmt(m: dict) -> str:
    return (f"GAUC {m['GAUC']:.4f} | nDCG@5 {m['nDCG@5']:.4f} | "
            f"primary {m['primary']:.4f}")


def train_model(model_name: str, data: Dataset, seed: int = config.DEFAULT_SEED,
                **model_kwargs):
    """Build + fit a model on an already-loaded Dataset. Returns the model."""
    model = build(model_name, seed=seed, **model_kwargs)
    model.fit(data)
    return model


def evaluate_model(model, data: Dataset, splits=("valid", "test")) -> dict:
    """Score a fitted model on the given splits. Returns {split: metrics}."""
    out = {}
    for split in splits:
        scores = model.predict(data, split)
        out[split] = evaluate(data.users(split), data.y(split), scores)
    return out


def run_experiment(model_name: str, data_dir=None, seed: int = config.DEFAULT_SEED,
                   log: bool = True, **model_kwargs) -> dict:
    """End-to-end: load, train, evaluate valid+test, print, optionally log."""
    data = load_dataset(data_dir or config.DATA_DIR)
    print(f"loaded: " + ", ".join(f"{k}={len(v):,d}" for k, v in data.splits.items()))

    model = train_model(model_name, data, seed=seed, **model_kwargs)
    res = evaluate_model(model, data)

    print(f"\n=== {model_name} (seed={seed}) ===")
    for split in ("valid", "test"):
        print(f"  {split:5s}  {_fmt(res[split])}")
    tp = res["test"]["primary"]
    print(f"\n  test primary {tp:.4f}  vs  FM {config.FM_BASELINE_PRIMARY}  "
          f"|  oracle {config.ORACLE_CEILING_PRIMARY}  "
          f"({100 * (tp - config.FM_BASELINE_PRIMARY):+.2f} pts vs FM)")

    if log:
        append_leaderboard(model_name, seed, res)
    return res


def append_leaderboard(model_name: str, seed: int, res: dict,
                       owner: str = "auto", note: str = "") -> None:
    """Append one measured run to results/leaderboard.md."""
    path = config.RESULTS_DIR / "leaderboard.md"
    va, te = res["valid"], res["test"]
    row = (f"| {date.today().isoformat()} | {owner} | {model_name} (seed={seed}) | "
           f"{va['GAUC']:.4f} | {va['nDCG@5']:.4f} | {va['primary']:.4f} | "
           f"{te['primary']:.4f} | {note} |\n")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(row)
