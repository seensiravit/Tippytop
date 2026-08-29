from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pytest

import tippytop.doctor as doctor_module
from tippytop.agent import summarize_dataset
from tippytop.config import RunConfig
from tippytop.doctor import run_doctor
from tippytop.experiments import predict_checkpoint
from tippytop.generated import GeneratedExperiment
from tippytop.runtime import ExperimentFailure, SandboxFailure, prepare_research_data, run_experiment
from tippytop.runtime.runner import _projected_full_seconds
from tippytop.starter import load_splits


SOURCE = """import numpy as np

def fit(train_rows, seed):
    grouped = train_rows.groupby("video_id")["long_view"].agg(["sum", "count"])
    rates = (grouped["sum"] / grouped["count"]).to_dict()
    global_mean = float(train_rows["long_view"].mean())
    return {"rates": rates, "global_mean": global_mean}, {"epochs_completed": 1}

def predict(model, rows):
    if "long_view" in rows.columns or "play_time_ms" in rows.columns:
        raise ValueError("prediction rows must not contain outcomes")
    return rows["video_id"].map(model["rates"]).fillna(model["global_mean"]).to_numpy(np.float32)
"""


def test_smoke_projection_allows_startup_but_scales_fit_work() -> None:
    assert _projected_full_seconds(4.0, 10_000, 1_000_000) == 4.0
    assert _projected_full_seconds(20.0, 10_000, 1_000_000) == pytest.approx(1505.0)

TABULAR_SOURCE = """from sklearn.ensemble import HistGradientBoostingClassifier
from tippytop.research import TabularEncoder, labels

def fit(train_rows, seed):
    encoder = TabularEncoder()
    features = encoder.fit_transform(train_rows)
    estimator = HistGradientBoostingClassifier(max_iter=5, random_state=seed)
    estimator.fit(features, labels(train_rows))
    return encoder, estimator

def predict(model, rows):
    encoder, estimator = model
    return estimator.predict_proba(encoder.transform(rows))[:, 1]
"""

INVALID_SCORE_SOURCE = """def fit(train_rows, seed):
    return None

def predict(model, rows):
    return []
"""

SYSTEM_EXIT_SOURCE = """def fit(train_rows, seed):
    raise SystemExit(0)

def predict(model, rows):
    return [0.0] * len(rows)
"""


def test_doctor_and_worker_smoke(synthetic_data_dir: Path, tmp_path: Path) -> None:
    config = RunConfig(
        data_dir=synthetic_data_dir,
        runs_dir=tmp_path / "runs",
        experiment_timeout=60,
        offline=True,
    )
    checks = run_doctor(config, check_llm=False)
    assert checks["ok"], checks["errors"]
    splits = load_splits(synthetic_data_dir)
    research_data = prepare_research_data(tmp_path / "research.pkl", synthetic_data_dir)
    experiment = GeneratedExperiment.from_dict(
        {
            "hypothesis": "worker smoke",
            "expected_effect": "produce a checkpoint",
            "source": SOURCE,
        }
    )
    result, output = run_experiment(
        config,
        experiment,
        tmp_path / "experiment",
        research_data,
    )
    checkpoint = Path(result["checkpoint"])
    assert checkpoint.is_absolute()
    assert Path(result["valid_scores"]).is_absolute()
    assert checkpoint.is_dir()
    with (checkpoint / "smoke-result.json").open(encoding="utf-8") as handle:
        smoke_result = json.load(handle)
    assert smoke_result["status"] == "ok"
    assert result["metrics"]["rows"] == 4
    assert "host validation primary=" in output["stdout"]
    scores = predict_checkpoint(checkpoint, splits, "test", data_dir=synthetic_data_dir)
    assert scores.shape == (4,)
    assert np.isfinite(scores).all()

    source_path = checkpoint / "experiment.py"
    source_path.write_text(source_path.read_text(encoding="utf-8") + "\n# tampered\n", encoding="utf-8")
    with pytest.raises(SandboxFailure, match="source hash"):
        predict_checkpoint(checkpoint, splits, "test", data_dir=synthetic_data_dir)


def test_online_doctor_requires_bubblewrap(
    synthetic_data_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = RunConfig(data_dir=synthetic_data_dir, runs_dir=tmp_path / "runs")
    monkeypatch.setattr(doctor_module.shutil, "which", lambda _: None)

    checks = run_doctor(config, check_llm=False)

    assert not checks["ok"]
    assert any("bubblewrap" in error for error in checks["errors"])


def test_smoke_failure_blocks_full_training(
    synthetic_data_dir: Path,
    tmp_path: Path,
) -> None:
    config = RunConfig(
        data_dir=synthetic_data_dir,
        runs_dir=tmp_path / "runs",
        experiment_timeout=60,
        offline=True,
    )
    splits = load_splits(synthetic_data_dir)
    research_data = prepare_research_data(tmp_path / "bad-research.pkl", synthetic_data_dir)
    experiment = GeneratedExperiment.from_dict(
        {
            "hypothesis": "return malformed scores",
            "expected_effect": "exercise the smoke gate",
            "source": INVALID_SCORE_SOURCE,
        }
    )
    artifact_dir = tmp_path / "bad-experiment"

    with pytest.raises(ExperimentFailure, match="expected "):
        run_experiment(config, experiment, artifact_dir, research_data)

    assert json.loads((artifact_dir / "smoke-result.json").read_text())["status"] == "error"
    assert not (artifact_dir / "request.json").exists()
    assert not (artifact_dir / "result.json").exists()


def test_generated_system_exit_cannot_forge_success(
    synthetic_data_dir: Path,
    tmp_path: Path,
) -> None:
    config = RunConfig(
        data_dir=synthetic_data_dir,
        runs_dir=tmp_path / "runs",
        experiment_timeout=60,
        offline=True,
    )
    research_data = prepare_research_data(tmp_path / "exit-research.pkl", synthetic_data_dir)
    experiment = GeneratedExperiment.from_dict(
        {
            "hypothesis": "exit before producing a model",
            "expected_effect": "exercise trusted result handling",
            "source": SYSTEM_EXIT_SOURCE,
        }
    )

    with pytest.raises(ExperimentFailure, match="SystemExit"):
        run_experiment(config, experiment, tmp_path / "exit-experiment", research_data)


def test_dataset_summary_excludes_test_labels(
    synthetic_splits: dict[str, list[tuple[object, ...]]],
) -> None:
    original = summarize_dataset(synthetic_splits)
    changed = dict(synthetic_splits)
    changed["test"] = [tuple((*row[:6], 1 - int(row[6]))) for row in synthetic_splits["test"]]
    assert summarize_dataset(changed) == original
    assert "test" not in original


def test_tabular_model_bundle_trains_and_replays(
    synthetic_data_dir: Path,
    tmp_path: Path,
) -> None:
    config = RunConfig(
        data_dir=synthetic_data_dir,
        runs_dir=tmp_path / "runs",
        experiment_timeout=60,
        offline=True,
    )
    splits = load_splits(synthetic_data_dir)
    research_data = prepare_research_data(tmp_path / "tabular-research.pkl", synthetic_data_dir)
    experiment = GeneratedExperiment.from_dict(
        {
            "hypothesis": "train a compact tabular model",
            "expected_effect": "exercise the generated research API",
            "source": TABULAR_SOURCE,
        }
    )

    result, _ = run_experiment(
        config,
        experiment,
        tmp_path / "tabular-experiment",
        research_data,
    )
    scores = predict_checkpoint(
        Path(result["checkpoint"]),
        splits,
        "test",
        data_dir=synthetic_data_dir,
    )

    assert scores.shape == (4,)
    assert np.isfinite(scores).all()


def test_research_artifact_excludes_test_split(
    synthetic_data_dir: Path,
    tmp_path: Path,
) -> None:
    summary_path = tmp_path / "research-schema.json"
    path = prepare_research_data(
        tmp_path / "research.pkl",
        synthetic_data_dir,
        summary_path=summary_path,
    )
    with path.open("rb") as handle:
        payload = pickle.load(handle)

    assert set(payload) == {"train", "valid"}
    assert "long_view" in payload["train"].columns
    assert "play_time_ms" in payload["train"].columns
    assert "long_view" not in payload["valid"].columns
    assert "play_time_ms" not in payload["valid"].columns
    assert len(payload["train"]) == 20
    assert len(payload["valid"]) == 4
    summary = json.loads(summary_path.read_text())
    assert summary["representation"] == "pandas.DataFrame"
    assert "long_view" in summary["training_only_columns"]
    assert "long_view" not in summary["prediction_columns"]
