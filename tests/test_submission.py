from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from tippytop.artifacts import RunStore, read_json, source_revision
from tippytop.config import RunConfig
from tippytop.submission import finalize_run, write_submission


def test_write_submission_preserves_row_order(tmp_path: Path) -> None:
    rows = [
        (20220429, "user-b", "video-z", "author", 0, 1000, 1),
        (20220429, "user-a", "video-y", "author", 1, 2000, 0),
    ]
    path = tmp_path / "submission.csv"

    write_submission(path, rows, [0.25, -1.5])

    with path.open(newline="", encoding="utf-8") as handle:
        written = list(csv.reader(handle))
    assert written == [
        ["row_id", "user_id", "video_id", "score"],
        ["0", "user-b", "video-z", "0.25"],
        ["1", "user-a", "video-y", "-1.5"],
    ]


def test_write_submission_rejects_nonfinite_scores(tmp_path: Path) -> None:
    rows = [(20220429, "user", "video", "author", 0, 1000, 1)]

    with pytest.raises(ValueError, match="not finite"):
        write_submission(tmp_path / "submission.csv", rows, [float("nan")])


def test_finalize_run_scores_test_once_and_reuses_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    store = RunStore(run_dir)
    config = RunConfig(data_dir=tmp_path / "data")
    test_rows = [
        (20220429, "user-b", "video-z", "author", 0, 1000, 1),
        (20220429, "user-a", "video-y", "author", 1, 2000, 0),
    ]
    state = {
        "source_revision": source_revision(),
        "baseline_valid": {"GAUC": 0.6, "nDCG@5": 0.5, "primary": 0.55},
        "best": {
            "id": "generated-1",
            "checkpoint": "checkpoint.npz",
            "metrics": {"GAUC": 0.7, "nDCG@5": 0.6, "primary": 0.65},
        },
        "iteration": 1,
        "stopping_reason": "converged",
        "test_evaluated": False,
    }
    calls = {"predict": 0}
    (run_dir / "checkpoint.npz").write_bytes(b"checkpoint")
    store.write_json("state.json", state)

    monkeypatch.setattr(
        "tippytop.submission.finalize.load_splits",
        lambda _data_dir: {"train": [], "valid": [], "test": test_rows},
    )

    def predict(*_args: object, **_kwargs: object) -> np.ndarray:
        calls["predict"] += 1
        return np.asarray([0.9, 0.1], dtype=np.float32)

    monkeypatch.setattr("tippytop.submission.finalize.predict_checkpoint", predict)
    monkeypatch.setattr(
        "tippytop.submission.finalize.evaluate",
        lambda *_args: {"GAUC": 0.8, "nDCG@5": 0.7, "primary": 0.75, "rows": 2, "users": 2},
    )
    monkeypatch.setattr(
        "tippytop.submission.finalize.validate_with_starter",
        lambda *_args: "Submission format OK",
    )

    first = finalize_run(store, config, state)
    second = finalize_run(store, config, state)

    assert calls["predict"] == 1
    assert first["reused"] is False
    assert second["reused"] is True
    assert state["test_evaluated"] is True
    assert read_json(run_dir / "results.json")["best_experiment"] == "generated-1"
    assert (run_dir / "final_submission.csv").is_file()
    assert (run_dir / "report.md").is_file()
    assert read_json(run_dir / "finalization.json")["status"] == "completed"
    assert read_json(run_dir / "submission_check.json")["rows"] == 2


def test_finalize_run_refuses_to_repeat_interrupted_test_evaluation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "checkpoint.npz").write_bytes(b"checkpoint")
    store = RunStore(run_dir)
    config = RunConfig(data_dir=tmp_path / "data")
    rows = [(20220429, "user", "video", "author", 0, 1000, 1)]
    state = {
        "source_revision": source_revision(),
        "best": {
            "id": "baseline-fm",
            "checkpoint": "checkpoint.npz",
            "metrics": {"GAUC": 0.6, "nDCG@5": 0.5, "primary": 0.55},
        },
        "test_evaluated": False,
    }
    store.write_json("state.json", state)
    calls = {"predict": 0, "evaluate": 0}

    monkeypatch.setattr(
        "tippytop.submission.finalize.load_splits",
        lambda _data_dir: {"train": [], "valid": [], "test": rows},
    )

    def predict(*_args: object, **_kwargs: object) -> np.ndarray:
        calls["predict"] += 1
        return np.asarray([0.5], dtype=np.float32)

    def interrupted_evaluate(*_args: object) -> dict[str, float]:
        calls["evaluate"] += 1
        raise RuntimeError("simulated crash after evaluation began")

    monkeypatch.setattr("tippytop.submission.finalize.predict_checkpoint", predict)
    monkeypatch.setattr("tippytop.submission.finalize.evaluate", interrupted_evaluate)

    with pytest.raises(RuntimeError, match="simulated crash"):
        finalize_run(store, config, state)
    with pytest.raises(RuntimeError, match="refusing to evaluate test again"):
        finalize_run(store, config, state)

    assert calls == {"predict": 1, "evaluate": 1}
    assert read_json(run_dir / "finalization.json")["status"] == "evaluation_started"
