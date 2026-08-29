# Tutorial — How to Work on Tippytop

A 5-minute practical guide. For the *why* behind the layout read
[`project-structure.md`](project-structure.md); for the goal read
[`../task.md`](../task.md).

---

## 0. The mental model (30 seconds)

Everything reduces to one line:

```
your model  →  scores (one float per eval row)  →  evaluate()  →  primary score
```

`evaluate()` is frozen (in the starter kit). Your job is to produce better
`scores`. Beat the FM baseline **primary 0.5946**; the ceiling is 0.8645.

---

## 1. One-time setup

```bash
# from the repo root
pip install -e ".[dev]"                 # installs numpy, pytest; enables `python -m tippytop`

# download the dataset into the vendored kit (~24 MB, no registration)
bash scripts/download_data.sh           # or:  powershell scripts/download_data.ps1
```

Verify the harness before trusting anything:

```bash
python -m tippytop run --model random --no-log     # test primary must be ~0.4754
python -m tippytop run --model fm    --no-log       # test primary must be ~0.595  (~50s)
```

> No install? Use the wrappers instead of `python -m tippytop`:
> `python scripts/run_experiment.py --model fm`

---

## 2. The interface (all you need day-to-day)

```bash
# train a model, evaluate valid & test, auto-log to results/leaderboard.md
python -m tippytop run    --model fm

# train + write a submission CSV for a split (auto-validates it)
python -m tippytop submit --model fm --split test --out results/submissions/fm_test.csv

# validate an existing CSV (format + alignment)
python -m tippytop check  results/submissions/fm_test.csv --split test

# validate + score an existing CSV (valid split only)
python -m tippytop score  results/submissions/pop_valid.csv --split valid
```

`--model` = `fm` | `pop` | `random` | *(anything you register)*.
FM flags: `--k`, `--lr`, `--epochs`, `--seed`.

---

## 3. Add your own model (the core workflow)

Say you want an FM trained with a ranking loss.

**1. Create `src/tippytop/models/fm_bpr.py`:**

```python
import numpy as np
from .base import Model
from ..data.dataset import Dataset
from . import register

@register("fm_bpr")               # this string is your --model name
class FMBpr(Model):
    name = "fm_bpr"

    def __init__(self, seed=0, **kw):
        self.seed = seed
        # ... hyperparams ...

    def fit(self, data: Dataset) -> "FMBpr":
        # train on data.enc["train"], early-stop on data.enc["valid"]
        # (X, y, users) = data.enc["train"]
        return self

    def predict(self, data: Dataset, split: str) -> np.ndarray:
        # return one float per row of data.splits[split], in row order
        ...
```

**2. Register it** — add to the import line at the bottom of
`src/tippytop/models/__init__.py`:

```python
from . import fm, popularity, random_model, fm_bpr   # add yours
```

**3. Run it** — no other file changes:

```bash
python -m tippytop run --model fm_bpr
```

That's the whole contract: `fit(data)` and `predict(data, split) -> scores`.
Everything else (loading, encoding, scoring, submission, logging) is shared.

> Model-agnostic loss functions go in `src/tippytop/losses/`, not inside a model,
> so other models can reuse them.

---

## 4. Measure honestly

- Iterate on **valid** primary; touch **test** only for final picks.
- A change is **real only if valid primary improves by > +0.002** (the noise band;
  FM's seed std is 0.0008). Smaller = noise.
- Log every run in [`../results/leaderboard.md`](../results/leaderboard.md) and drop
  a config in `experiments/configs/` so it's reproducible.

---

## 5. Team workflow

1. Work on your own branch: `git checkout -b dev/<name>` (e.g. `dev/alice`).
2. Work only in your files (models are one-file-each, so no collisions).
3. Run `pytest tests/ -v` (harness sanity) before pushing.
4. PR into `main`; put your best valid/test primary in the description.

---

## 6. The rules (don't break these)

| ✅ Do | ❌ Don't |
|---|---|
| Import the kit via `tippytop.kit` | Edit anything in `kuairand-starter-kit/` |
| Report **valid** primary while iterating | Tune against **test** |
| Log runs in `results/leaderboard.md` | Trust a Δ < 0.002 "improvement" |
| Validate with `check` before submitting | Hand-edit submission CSVs |

---

## 7. Troubleshooting

| Symptom | Fix |
|---|---|
| `No module named tippytop` | `pip install -e .`, or use `python scripts/run_experiment.py ...` |
| `Vendored starter kit not found` | you're not at the repo root / kit folder moved |
| data errors | run the download script; data lives in `kuairand-starter-kit/kuairand-starter-kit/KuaiRand-Pure/data` |
| `random` ≠ ~0.475 | the harness is broken — stop and fix before trusting results |
