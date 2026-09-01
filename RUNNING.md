# How to test this

The problem statement specifies the benchmark: **KuaiRand-Pure**, which alone
determines 100% of the primary score (KuaiRand-1k / 27k are bonus only). It is a
~46 MB download from Zenodo, no registration.

## 0. Install the package first

Nothing below works until `tippytop` is importable — `No module named tippytop`
means this step was skipped.

**Windows / PowerShell**

    python -m pip install -e .

**macOS / Linux**

    pip install -e .

**With uv** (what `CLAUDE.md` specifies, and it manages the interpreter for you)

    uv sync

> **Python version.** `pyproject.toml` requires >= 3.11. Python **3.14** is new
> enough that numpy or LightGBM may not publish wheels for it yet; if the install
> fails while building numpy, that is the cause. Use 3.11 or 3.12:
> `py -3.12 -m pip install -e .` on Windows. `uv sync` sidesteps this entirely.

## 1. Get the data (once)

**Windows / PowerShell** — there is no `bash` on a stock Windows box, so use the
`.ps1`. (`bash scripts/download_data.sh` fails with a WSL relay error.)

    .\scripts\download_data.ps1

**macOS / Linux**

    bash scripts/download_data.sh

Lands in `./KuaiRand-Pure/data/`, which is the default `--data_dir` everywhere.
The download is ~46 MB.

## 2. Run the organizers' sanity check FIRST

The starter kit is explicit that this comes before anything else — if it fails,
the harness is broken and no other number can be trusted:

    python -m tippytop run --model random --no-log
    # uv:  uv run python -m tippytop run --model random --no-log

Expect **test primary ≈ 0.4753 (±0.001)**. Your `CLAUDE.md` records 0.4754
verified, so this should reproduce.

## 3. Confirm the baseline you have to beat

    python -m tippytop run --model fm --no-log      # ~40-60 s, CPU

Expect valid ≈ 0.6015, test ≈ 0.5946. Note the 10-seed mean is **0.6015** — seed
42's 0.6019 is a full sigma high and should not be quoted as "the baseline".

## 4. Phase 0 — measure the instrument (no training beyond two models)

    python examples/measure_first.py --data_dir KuaiRand-Pure/data

(Windows: forward slashes work fine in the `--data_dir` value.)

Prints four things:

1. whether the split can move the metric at all (`diagnose_eval_split`),
2. a **paired bootstrap CI** on a real comparison — `fm_listwise` vs `fm`,
3. how many seeds a claimed effect of a given size actually needs,
4. what best-of-N selection manufactures from noise alone.

## 5. Run a real experimental block

```python
from tippytop.data.dataset import load_dataset
from tippytop.training.runner import train_model
from tippytop.experiments import Candidate, run_block

d = load_dataset("KuaiRand-Pure/data")
u, y = d.users("valid"), d.y("valid")
model = lambda name, **kw: (lambda seed:
    train_model(name, d, seed=seed, verbose=False, **kw).predict(d, "valid"))

r = run_block(
    "does a ranking objective beat pointwise at matched batching?",
    [Candidate("fm_listwise", model("fm_listwise")),
     Candidate("fm_hybrid",   model("fm_hybrid"))],
    Candidate("fm", model("fm")),
    u, y, screen_seeds=1, confirm_seeds=3)

print(r.summary())      # effect size + both intervals + accept/reject
```

## 6. The experiment worth running first

Per the analysis, this decides between two competing explanations for the
-0.0023 batching penalty, and they predict opposite outcomes:

```python
r = run_block(
    "is the batching penalty a batch-size artifact?",
    [Candidate(f"ctrl_lr{lr}", model("fm_hybrid", alpha=1.0, list_size=32,
                                     groups_per_batch=256, lr=lr))
     for lr in (5e-5, 1e-4, 2e-4, 5e-4, 1e-3)],
    Candidate("fm", model("fm")),
    u, y, screen_seeds=1, confirm_seeds=3)
```

`list_size=32 x groups_per_batch=256` is exactly 8,192 rows per batch, matching
the baseline. The lr bracket spans **both directions** deliberately: the Surge
Phenomenon result (NeurIPS 2024) shows optimal lr for Adam is non-monotonic in
batch size, so the optimum may sit below your tested range rather than above it.

## Unit tests (no data needed)

    python -m pytest tests/ -q          # 201 tests

---

## Measuring the item-aggregate direction

    python examples/measure_aggregates.py --data_dir KuaiRand-Pure/data

`data/aggregates.py` is the one direction the organisers' ablation does not cover
(`ablation_features.py` tests four categorical IDs, all redundant given
`video_id`; continuous engagement rates were never tested). It is already
implemented, train-only, with leave-one-out so a row cannot predict itself.

What has not been established is **how much of the FFM win comes from
field-awareness and how much from the aggregate features** — they are currently
only reachable inside `ffm` and `lgbm_rank`, so the two effects are confounded.
`+0.0009 valid / +0.0019 test` currently supports "FFM beats FM", not "aggregates
beat no aggregates".

The script reports which aggregate features actually vary *within* a user (the
only ones that can reorder anything), then runs a screened, seed-confirmed block
against an FM control with a paired-bootstrap interval.

To fully separate the two claims you need one more arm: FFM with the aggregate
columns zeroed. That is the same control discipline that produced the batching
finding, and it converts a confounded result into a clean one.
