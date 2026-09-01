# models/

One file per model, each subclassing `Model` (`base.py`) and registered via
`@register("name")` in `__init__.py`.

The contract is two methods — `fit(data: Dataset)` and
`predict(data: Dataset, split: str) -> np.ndarray` (one float per row, in row
order). Everything else (loading, encoding, scoring, submission) is shared, so
a new model never touches the data or eval code.

| File | Model name(s) | Notes |
|---|---|---|
| `fm.py` | `fm` | FM baseline (k=16, lr=0.001, 5 fields). Reference: primary 0.5946 test. |
| `ffm.py` | `ffm` | Field-aware FM (k=4). +0.0009 valid / +0.0019 test over FM (6-seed mean). |
| `fm_rank.py` | `fm_listwise`, `fm_bpr`, `fm_hybrid` | FM with ranking objectives. Better objective, cannot repay grouped-batching cost. See leaderboard. |
| `fm_multitask.py` | `fm_multitask` | FM + auxiliary heads on `is_click` and D2Q watch-time. Closed (6/6 vs control, monotone wrong direction). |
| `ensemble.py` | `fm_seedavg`, `fm_blend`, `fm_diverse` | Rank-averaged ensembles. Best: 6×FM+6×FFM, valid 0.6045 / test 0.5976. |
| `lgbm_rank.py` | `lgbm_rank` | LightGBM LambdaRank. Closed — no user×item signal for axis-aligned splits (valid 0.5887). |
| `popularity.py` | `pop` | Item-popularity baseline. primary 0.5715. |
| `random_model.py` | `random` | Harness sanity check. Must score primary ≈ 0.4753. |

## Adding a model

1. New file `src/tippytop/models/<name>.py`:

```python
import numpy as np
from .base import Model
from ..data.dataset import Dataset
from . import register

@register("mymodel")
class MyModel(Model):
    name = "mymodel"

    def fit(self, data: Dataset) -> "MyModel":
        # data.splits["train"], data.enc["train"], data.dim, ...
        return self

    def predict(self, data: Dataset, split: str) -> np.ndarray:
        # one float per row of data.splits[split], in row order
        ...
```

2. Add the module to the import line at the bottom of `__init__.py`.
3. `uv run python -m tippytop run --model mymodel`. No other changes.

Model-agnostic losses go in `../losses/`, not inside a model.
