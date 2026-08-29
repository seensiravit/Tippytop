"""Harness sanity checks. Run once data is downloaded:

    python -m pytest tests/ -v

The critical one: `random` must score primary ~= 0.475, else the eval harness is
broken and no result can be trusted (per README).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tippytop import config  # noqa: E402
from tippytop.kit import DEFAULT_DATA_DIR  # noqa: E402

import pytest  # noqa: E402

pytestmark = pytest.mark.skipif(
    not (DEFAULT_DATA_DIR / "log_standard_4_08_to_4_21_pure.csv").exists(),
    reason="KuaiRand-Pure data not downloaded (run scripts/download_data).",
)


def test_random_sanity():
    """random baseline reproduces primary ~= 0.475 (+/- 0.001)."""
    import numpy as np
    from tippytop.kit import load, encode, evaluate

    splits = load(str(DEFAULT_DATA_DIR))
    enc, _ = encode(splits)
    _, y, users = enc["test"]
    rng = np.random.default_rng(0)
    scores = rng.random(len(y))
    m = evaluate(users, y, scores)
    assert abs(m["primary"] - config.RANDOM_SANITY_PRIMARY) < 0.005
