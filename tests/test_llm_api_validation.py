from __future__ import annotations

import pytest

from tippytop.llm.api_validation import validate_installed_api_calls


INVALID = """import lightgbm as lgb

def fit(train_rows, seed):
    model = lgb.LGBMRanker(objective="lambdarank", n_estimators=10)
    params = {"objective": "lambdarank", "metric": "ndcg", "learning_rate": 0.1}
    model.fit([[0.0], [1.0]], [0, 1], group=[2], **params)
    return model

def predict(model, rows):
    return model.predict(rows)
"""


VALID = """import lightgbm as lgb

def fit(train_rows, seed):
    model = lgb.LGBMRanker(
        objective="lambdarank", metric="ndcg", learning_rate=0.1, n_estimators=10
    )
    model.fit([[0.0], [1.0]], [0, 1], group=[2])
    return model

def predict(model, rows):
    return model.predict(rows)
"""


def test_expanded_fit_parameters_are_checked_against_installed_api() -> None:
    with pytest.raises(ValueError) as captured:
        validate_installed_api_calls(INVALID)

    message = str(captured.value)
    assert "learning_rate" in message
    assert "metric" in message
    assert "objective" in message
    assert 'experiment.py", line 6' in message


def test_constructor_parameters_do_not_trigger_fit_validation() -> None:
    validate_installed_api_calls(VALID)


def test_ranker_rejects_statically_resolvable_constant_labels() -> None:
    source = """import numpy as np
import lightgbm as lgb

def fit(train_rows, seed):
    model = lgb.LGBMRanker(objective="lambdarank")
    labels = np.ones(2)
    model.fit([[0.0], [1.0]], labels, group=[2])
    return model

def predict(model, rows):
    return model.predict(rows)
"""

    with pytest.raises(ValueError) as captured:
        validate_installed_api_calls(source)

    assert "non-constant relevance labels" in str(captured.value)


def test_reassigned_or_different_scope_receiver_is_not_stale_estimator() -> None:
    source = """import lightgbm as lgb

class Wrapper:
    def fit(self, **kwargs):
        return self

def first():
    model = lgb.LGBMRanker()
    model = Wrapper()
    model.fit(objective="custom wrapper option")

def second():
    model = Wrapper()
    model.fit(metric="custom wrapper option")

def fit(train_rows, seed):
    return Wrapper()

def predict(model, rows):
    return [0.5] * len(rows)
"""

    validate_installed_api_calls(source)
