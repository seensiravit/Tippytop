"""Final test scoring and organizer submission validation."""

from __future__ import annotations

import fcntl
import hashlib
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np

from ..artifacts import (
    RunStore,
    assert_source_revision,
    atomic_save_npy,
    atomic_write_text,
    read_json,
    utc_now,
)
from ..config import RunConfig
from ..experiments import predict_checkpoint
from ..starter import evaluate, load_splits
from .checker import validate_with_starter, write_submission
from .report import render_report


def finalize_run(store: RunStore, config: RunConfig, state: dict[str, Any]) -> dict[str, Any]:
    with _finalization_lock(store.path / ".finalization.lock"):
        durable_state = read_json(store.path / "state.json") if (store.path / "state.json").is_file() else state
        state.clear()
        state.update(durable_state)
        return _finalize_locked(store, config, state)


def _finalize_locked(store: RunStore, config: RunConfig, state: dict[str, Any]) -> dict[str, Any]:
    finalized_started = time.monotonic()
    submission_path = store.path / "final_submission.csv"
    # Finalization is idempotent so resume/submit cannot score test a second time.
    if state.get("test_evaluated"):
        if not submission_path.is_file():
            raise RuntimeError("state says test was evaluated but final_submission.csv is missing")
        validation_output = validate_with_starter(submission_path, config.data_dir)
        return {
            "submission": store.relative_path(submission_path),
            "test_metrics": state["test_metrics"],
            "validation_output": validation_output,
            "reused": True,
        }

    best = state.get("best")
    if not best or not best.get("checkpoint"):
        raise RuntimeError("run has no selected checkpoint")
    assert_source_revision(str(state["source_revision"]))
    checkpoint = store.resolve_path(best["checkpoint"])
    checkpoint_hash = _checkpoint_hash(checkpoint)
    transaction_path = store.path / "finalization.json"
    transaction = read_json(transaction_path) if transaction_path.is_file() else None
    prediction_started_now = transaction is None
    if transaction is None:
        transaction = {
            "version": 1,
            "status": "prediction_started",
            "started_at": utc_now(),
            "selected_experiment": best["id"],
            "checkpoint": best["checkpoint"],
            "checkpoint_sha256": checkpoint_hash,
        }
        store.write_json("finalization.json", transaction)
    elif transaction.get("selected_experiment") != best["id"] or transaction.get(
        "checkpoint_sha256"
    ) != checkpoint_hash:
        raise RuntimeError("finalization transaction does not match the selected checkpoint")

    splits = load_splits(config.data_dir)
    scores_path = store.path / "final_test_scores.npy"
    if transaction["status"] == "prediction_started":
        if scores_path.is_file():
            test_scores = np.load(scores_path, allow_pickle=False)
        elif prediction_started_now:
            test_scores = predict_checkpoint(checkpoint, splits, "test", data_dir=config.data_dir)
            atomic_save_npy(scores_path, np.asarray(test_scores, dtype=np.float32))
        else:
            raise RuntimeError(
                "test prediction was interrupted before durable scores were written; refusing to "
                "predict test again"
            )
        transaction.update(
            status="predictions_ready",
            scores="final_test_scores.npy",
            scores_sha256=_file_sha256(scores_path),
        )
        store.write_json("finalization.json", transaction)
    else:
        if not scores_path.is_file() or transaction.get("scores_sha256") != _file_sha256(scores_path):
            raise RuntimeError("durable final test scores are missing or changed")
        test_scores = np.load(scores_path, allow_pickle=False)

    test_rows = splits["test"]
    evaluation_started_now = transaction["status"] == "predictions_ready"
    if evaluation_started_now:
        transaction["status"] = "evaluation_started"
        store.write_json("finalization.json", transaction)
        test_metrics = evaluate(
            [row[1] for row in test_rows],
            [row[6] for row in test_rows],
            test_scores,
        )
        normalized_metrics = {
            key: int(value) if key in {"users", "rows"} else float(value)
            for key, value in test_metrics.items()
        }
        transaction.update(status="evaluated", test_metrics=normalized_metrics)
        store.write_json("finalization.json", transaction)
    elif transaction["status"] in {"evaluated", "checked", "completed"}:
        normalized_metrics = dict(transaction["test_metrics"])
    else:
        raise RuntimeError(
            "test evaluation was interrupted before metrics were durable; refusing to evaluate test again"
        )

    write_submission(submission_path, test_rows, test_scores)
    validation_output = validate_with_starter(submission_path, config.data_dir)
    submission_check = {
        "checker_output": validation_output,
        "rows": len(test_rows),
        "selected_experiment": best["id"],
        "checkpoint_sha256": checkpoint_hash,
        "submission_sha256": _file_sha256(submission_path),
    }
    store.write_json("submission_check.json", submission_check)
    transaction.update(status="checked", submission_check=submission_check)
    store.write_json("finalization.json", transaction)

    finalization_seconds = time.monotonic() - finalized_started
    search_seconds = float(state.get("elapsed_seconds", 0.0))
    state["test_evaluated"] = True
    state["test_evaluated_at"] = utc_now()
    state["test_metrics"] = normalized_metrics
    state["submission"] = store.relative_path(submission_path)
    state["status"] = "completed"
    state["search_elapsed_seconds"] = search_seconds
    state["finalization_elapsed_seconds"] = finalization_seconds
    state["elapsed_seconds"] = search_seconds + finalization_seconds
    store.write_json("state.json", state)
    results = {
        "baseline_valid": state.get("baseline_valid"),
        "best_validation": best["metrics"],
        "best_experiment": best["id"],
        "test": normalized_metrics,
        "submission": store.relative_path(submission_path),
        "stopping_reason": state.get("stopping_reason"),
        "iterations": state.get("iteration", 0),
        "llm_usage": state.get("llm_usage", {}),
        "search_elapsed_seconds": search_seconds,
        "finalization_elapsed_seconds": finalization_seconds,
        "elapsed_seconds": state["elapsed_seconds"],
        "manual_interventions": state.get("manual_interventions", 0),
        "gpu_hours": 0,
    }
    store.write_json("results.json", results)
    atomic_write_text(store.path / "report.md", render_report(results))
    transaction.update(status="completed", completed_at=utc_now())
    store.write_json("finalization.json", transaction)
    store.event("run_finalized", test_metrics=normalized_metrics, submission=str(submission_path))
    return {
        "submission": store.relative_path(submission_path),
        "test_metrics": normalized_metrics,
        "validation_output": validation_output,
        "reused": False,
    }
@contextmanager
def _finalization_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_hash(path: Path) -> str:
    if path.is_file():
        return _file_sha256(path)
    if not path.is_dir():
        raise RuntimeError(f"selected checkpoint is missing: {path}")
    digest = hashlib.sha256()
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(child.relative_to(path).as_posix().encode("utf-8"))
        digest.update(_file_sha256(child).encode("ascii"))
    return digest.hexdigest()
