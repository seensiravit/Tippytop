from __future__ import annotations

import hashlib
from dataclasses import FrozenInstanceError

import pytest

from tippytop.generated import GeneratedExperiment, MAX_SOURCE_LENGTH, parse_json_object


VALID_SOURCE = '''"""A small CPU-only ranking experiment."""
from dataclasses import dataclass
from math import log1p

import numpy as np
from sklearn.linear_model import LogisticRegression

FEATURE_COLUMNS = ("watch_ratio", "likes")
OPTIONS: dict[str, int] = {"max_iter": 50}


@dataclass
class Batch:
    values: np.ndarray


def _features(rows):
    return np.asarray(
        [[log1p(max(float(row[column]), 0.0)) for column in FEATURE_COLUMNS] for row in rows],
        dtype=float,
    )


def fit(train_rows, seed):
    x = _features(train_rows)
    y = np.asarray([int(row["label"]) for row in train_rows])
    model = LogisticRegression(random_state=seed, **OPTIONS)
    model.fit(x, y)
    return model


def predict(model, rows):
    return model.predict_proba(_features(rows))[:, 1]
'''


def experiment(source: str = VALID_SOURCE) -> GeneratedExperiment:
    return GeneratedExperiment.from_dict(
        {
            "hypothesis": "A linear ranker should improve calibration.",
            "expected_effect": "Lower validation log loss.",
            "source": source,
        }
    )


def test_valid_experiment_is_immutable_and_has_stable_schema_and_hash() -> None:
    generated = experiment()

    assert generated.to_dict() == {
        "hypothesis": "A linear ranker should improve calibration.",
        "expected_effect": "Lower validation log loss.",
        "source": VALID_SOURCE,
    }
    assert generated.source_hash == hashlib.sha256(VALID_SOURCE.encode("utf-8")).hexdigest()
    assert GeneratedExperiment.from_dict(generated.to_dict()) == generated
    with pytest.raises(FrozenInstanceError):
        generated.source = "changed"  # type: ignore[misc]


@pytest.mark.parametrize(
    "change, message",
    [
        ({"expected_effect": None}, "expected_effect"),
        ({"hypothesis": "  "}, "hypothesis"),
        ({"source": 123}, "source"),
        ({"extra": "no"}, "unknown fields"),
    ],
)
def test_from_dict_rejects_invalid_schema(change: dict[str, object], message: str) -> None:
    value: dict[str, object] = experiment().to_dict()
    value.update(change)
    with pytest.raises(ValueError, match=message):
        GeneratedExperiment.from_dict(value)  # type: ignore[arg-type]

    if "extra" in change:
        value.pop("extra")
        value.pop("source")
    else:
        value.pop(next(iter(change)))
    with pytest.raises(ValueError, match="missing fields"):
        GeneratedExperiment.from_dict(value)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "source, message",
    [
        ("def fit(train_rows, seed):\n    return None\n", "predict"),
        (
            "def fit(train_rows):\n    return None\n\ndef predict(model, rows):\n    return []\n",
            "fit must have exactly",
        ),
        (
            "def fit(train_rows, seed, extra=0):\n    return None\n\ndef predict(model, rows):\n    return []\n",
            "fit must have exactly",
        ),
        (
            "def fit(train_rows, seed):\n  return None\n def predict(model, rows):\n  return []\n",
            "valid Python",
        ),
    ],
)
def test_rejects_malformed_source_and_missing_or_invalid_signatures(
    source: str, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        experiment(source)


def test_accepts_equivalent_entry_point_parameter_names() -> None:
    generated = experiment(
        "def fit(rows, random_seed=42):\n    return None\n\n"
        "def predict(model_bundle, impressions=()):\n    return []\n"
    )

    assert "model_bundle" in generated.source


def test_rejects_unreasonably_long_source() -> None:
    with pytest.raises(ValueError, match="character limit"):
        experiment(" " * MAX_SOURCE_LENGTH + VALID_SOURCE)


@pytest.mark.parametrize(
    "unsafe_line, message",
    [
        ("import os", "import from 'os'"),
        ("import subprocess", "import from 'subprocess'"),
        ("from pathlib import Path", "import from 'pathlib'"),
        ("from tippytop.worker import run", "tippytop.worker"),
        ("from .models import LinearModel", "relative imports"),
        ("handle = open('/tmp/result', 'w')", "call to open"),
        ("file_open = open", "reference to open"),
        ("result = eval('1 + 1')", "call to eval"),
        ("result = getattr(model, 'coef_')", "call to getattr"),
        ("result = model.__class__", "may not begin"),
        ("result = globals()", "call to globals"),
    ],
)
def test_rejects_filesystem_subprocess_introspection_and_import_behavior(
    unsafe_line: str, message: str
) -> None:
    source = f'''def fit(train_rows, seed):
    {unsafe_line}
    return None


def predict(model, rows):
    return []
'''
    with pytest.raises(ValueError, match=message):
        experiment(source)


def test_json_extraction_accepts_one_object_and_optional_fence() -> None:
    assert parse_json_object('{"hypothesis": "test"}') == {"hypothesis": "test"}
    assert parse_json_object('```json\n{"hypothesis": "test"}\n```') == {
        "hypothesis": "test"
    }
    assert parse_json_object('Result:\n{"hypothesis": "test"}') == {"hypothesis": "test"}


def test_data_science_and_safe_standard_imports_are_available() -> None:
    source = """import json
import pickle

import pandas as pd

def fit(train_rows, seed):
    return pickle.loads(pickle.dumps(pd.DataFrame(train_rows)))

def predict(model, rows):
    return json.loads("[0.0]") * len(rows)
"""
    assert experiment(source).source == source


def test_normal_top_level_model_configuration_is_available() -> None:
    source = """from typing import Any, Tuple

ModelBundle = Tuple[Any, Any]
OPTIONS = dict(max_iter=50)

def fit(train_rows, seed):
    return OPTIONS

def predict(model, rows):
    return [0.0] * len(rows)
"""
    assert experiment(source).source == source


def test_normal_class_initializers_are_available() -> None:
    source = """class Model:
    def __init__(self):
        self.value = 1.0

def fit(train_rows, seed):
    return Model()

def predict(model, rows):
    return [model.value] * len(rows)
"""
    assert experiment(source).source == source


@pytest.mark.parametrize(
    "response, message",
    [
        ('{"a": 1} explanation', "after the JSON object"),
        ('{"a": 1} {"b": 2}', "after the JSON object"),
        ("not JSON", "does not contain valid JSON"),
        ("[1, 2, 3]", "does not contain valid JSON"),
    ],
)
def test_json_extraction_rejects_trailing_or_non_object_content(
    response: str, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        parse_json_object(response)
