# models/

One file per model, each subclassing `Model` (`base.py`) and registered in
`__init__.py` via `@register("name")`.

The contract is tiny — `fit(enc, dim)` and `predict(X) -> scores`. Everything
else (loading, encoding, scoring, submission) is shared, so a new model never
touches the data or eval code.

| File | Model | Owner |
|---|---|---|
| `fm.py` | FM baseline adapter (0.5946) | _shared reference_ |
| `fm_bpr.py` | FM + pairwise BPR loss | _TBD_ |
| `fm_listwise.py` | FM + per-user softmax loss | _TBD_ |
| `din.py` | Sequence / interest model | _TBD_ |
| `deepfm.py` | DeepFM / DCN | _TBD_ |

Losses that are model-agnostic live in `../losses/`, not here.
