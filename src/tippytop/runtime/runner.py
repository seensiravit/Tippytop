"""Trusted baseline and sandboxed generated-experiment subprocess runners."""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from ..artifacts import (
    HostRevisionChanged,
    assert_source_revision,
    atomic_write_json,
    atomic_write_text,
    read_json,
)
from ..config import RunConfig
from ..generated import GeneratedExperiment
from ..research.data import load_prediction_frame
from ..starter import evaluate, load_splits
from .sandbox import SandboxFailure, predict_generated, run_worker_sandboxed


class ExperimentFailure(RuntimeError):
    pass


def run_reference_baseline(
    config: RunConfig,
    artifact_dir: Path,
) -> tuple[dict[str, Any], dict[str, str]]:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    request_path = artifact_dir / "request.json"
    result_path = artifact_dir / "result.json"
    atomic_write_json(
        request_path,
        {
            "mode": "reference_baseline",
            "data_dir": str(config.data_dir.resolve()),
            "artifact_dir": str(artifact_dir.resolve()),
            "seed": config.seed,
        },
    )
    return _run_trusted_worker(request_path, result_path, config.experiment_timeout)


def run_experiment(
    config: RunConfig,
    experiment: GeneratedExperiment,
    artifact_dir: Path,
    research_data_path: Path,
    *,
    timeout: int | None = None,
    run_root: Path | None = None,
    expected_source_revision: str | None = None,
) -> tuple[dict[str, Any], dict[str, str]]:
    try:
        return _run_experiment(
            config,
            experiment,
            artifact_dir,
            research_data_path,
            timeout=timeout,
            run_root=run_root,
            expected_source_revision=expected_source_revision,
        )
    except (ExperimentFailure, HostRevisionChanged):
        raise
    except Exception as error:
        raise ExperimentFailure(
            f"trusted experiment validation failed: {type(error).__name__}: {error}"
        ) from error


def _run_experiment(
    config: RunConfig,
    experiment: GeneratedExperiment,
    artifact_dir: Path,
    research_data_path: Path,
    *,
    timeout: int | None,
    run_root: Path | None,
    expected_source_revision: str | None,
) -> tuple[dict[str, Any], dict[str, str]]:
    if expected_source_revision is not None:
        assert_source_revision(expected_source_revision)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    source_path = artifact_dir / "experiment.py"
    atomic_write_text(source_path, experiment.source)
    smoke_request_path = artifact_dir / "smoke-request.json"
    smoke_result_path = artifact_dir / "smoke-result.json"
    smoke_result_path.unlink(missing_ok=True)
    request = {
        "data_path": str(research_data_path.resolve()),
        "source_path": str(source_path.resolve()),
        "artifact_dir": str(artifact_dir.resolve()),
        "seed": config.seed,
        "hypothesis": experiment.hypothesis,
        "expected_effect": experiment.expected_effect,
        "source_hash": experiment.source_hash,
    }
    atomic_write_json(smoke_request_path, {**request, "mode": "generated_smoke"})
    execution_timeout = min(config.experiment_timeout, timeout or config.experiment_timeout)
    deadline = time.monotonic() + execution_timeout
    smoke_started = time.monotonic()
    try:
        smoke = run_worker_sandboxed(
            smoke_request_path,
            smoke_result_path,
            writable_dir=artifact_dir,
            readonly_paths=[research_data_path.resolve()],
            timeout=min(_remaining_seconds(deadline), 300),
        )
    except SandboxFailure as error:
        raise ExperimentFailure(f"generated smoke check failed: {error}") from error
    smoke_duration = time.monotonic() - smoke_started
    smoke_output = {"stdout": smoke.stdout, "stderr": smoke.stderr}
    smoke_result, _ = _validated_result(smoke.returncode, smoke_result_path, smoke_output)
    projected_seconds = _projected_full_seconds(
        smoke_duration,
        int(smoke_result.get("train_rows", 0)),
        int(smoke_result.get("available_train_rows", 0)),
    )
    remaining_seconds = _remaining_seconds(deadline)
    if smoke_duration >= 15 and projected_seconds > remaining_seconds:
        raise ExperimentFailure(
            "generated smoke fit is not full-data feasible: "
            f"{smoke_duration:.1f}s for {smoke_result.get('train_rows')} sampled rows projects to "
            f"at least {projected_seconds:.1f}s for {smoke_result.get('available_train_rows')} rows, "
            f"exceeding the remaining {remaining_seconds}s experiment budget"
        )
    if source_path.read_text(encoding="utf-8") != experiment.source:
        raise ExperimentFailure("generated experiment modified its persisted source during smoke")
    if expected_source_revision is not None:
        assert_source_revision(expected_source_revision)

    request_path = artifact_dir / "request.json"
    result_path = artifact_dir / "result.json"
    for stale in (result_path, artifact_dir / "model.pkl", artifact_dir / "valid_scores.npy"):
        stale.unlink(missing_ok=True)
    atomic_write_json(request_path, {**request, "mode": "generated_train"})
    started = time.monotonic()
    try:
        completed = run_worker_sandboxed(
            request_path,
            result_path,
            writable_dir=artifact_dir,
            readonly_paths=[research_data_path.resolve()],
            timeout=_remaining_seconds(deadline),
        )
    except SandboxFailure as error:
        raise ExperimentFailure(str(error)) from error
    duration = time.monotonic() - started
    output = {
        "stdout": smoke.stdout + completed.stdout,
        "stderr": smoke.stderr + completed.stderr,
    }
    if source_path.read_text(encoding="utf-8") != experiment.source:
        raise ExperimentFailure("generated experiment modified its own persisted source")
    worker_result, _ = _validated_result(completed.returncode, result_path, output)
    if expected_source_revision is not None:
        assert_source_revision(expected_source_revision)

    model_path = artifact_dir / "model.pkl"
    scores_path = artifact_dir / "valid_scores.npy"
    if not model_path.is_file() or not scores_path.is_file():
        raise ExperimentFailure("generated worker exited without a model and validation scores")
    valid_rows = load_splits(config.data_dir)["valid"]
    scores = np.load(scores_path, allow_pickle=False)
    if scores.shape != (len(valid_rows),) or not np.isfinite(scores).all():
        raise ExperimentFailure("generated experiment persisted invalid validation scores")
    score_diagnostics = _score_diagnostics(scores)
    if len(scores) >= 100 and score_diagnostics["unique_scores"] <= 2:
        raise ExperimentFailure(
            "generated experiment returned only class labels; ranking requires continuous "
            "probabilities, decision scores, regression values, or logits"
        )

    manifest = {
        "format": "tippytop-generated-v2",
        "source_hash": experiment.source_hash,
        "hypothesis": experiment.hypothesis,
        "expected_effect": experiment.expected_effect,
        "metadata": worker_result.get("metadata", {}),
        "score_diagnostics": score_diagnostics,
    }
    atomic_write_json(artifact_dir / "manifest.json", manifest)
    replayed = predict_generated(
        artifact_dir,
        load_prediction_frame(config.data_dir, "valid"),
        timeout=_remaining_seconds(deadline),
    )
    if not np.array_equal(scores, replayed):
        raise ExperimentFailure("serialized generated checkpoint does not replay its validation scores")
    if expected_source_revision is not None:
        assert_source_revision(expected_source_revision)

    # Validation truth stays in this trusted host and is never mounted into the sandbox.
    metrics = evaluate(
        [str(row[1]) for row in valid_rows],
        [int(row[6]) for row in valid_rows],
        scores,
    )
    normalized_metrics = {
        key: int(value) if key in {"users", "rows"} else float(value)
        for key, value in metrics.items()
    }
    result = {
        "metrics": normalized_metrics,
        "model_type": "generated",
        "checkpoint": _stored_path(artifact_dir, run_root),
        "valid_scores": _stored_path(scores_path, run_root),
        "history": worker_result.get("history", []),
        "epochs_completed": int(worker_result.get("epochs_completed", 0)),
        "pairs": int(worker_result.get("pairs", 0)),
        "duration_seconds": duration,
        "metadata": {
            **dict(worker_result.get("metadata", {})),
            "score_diagnostics": score_diagnostics,
        },
    }
    atomic_write_json(result_path, {"status": "ok", "result": result})
    output["stdout"] += f"host validation primary={result['metrics']['primary']:.5f}\n"
    return result, output


def _run_trusted_worker(
    request_path: Path,
    result_path: Path,
    timeout: int,
) -> tuple[dict[str, Any], dict[str, str]]:
    command = [
        sys.executable,
        "-m",
        "tippytop.runtime.worker",
        str(request_path.resolve()),
        str(result_path.resolve()),
    ]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        raise ExperimentFailure(f"baseline timed out after {timeout}s") from error
    output = {"stdout": completed.stdout, "stderr": completed.stderr}
    return _validated_result(completed.returncode, result_path, output)


def _validated_result(
    returncode: int,
    result_path: Path,
    output: dict[str, str],
) -> tuple[dict[str, Any], dict[str, str]]:
    if not result_path.is_file():
        raise ExperimentFailure(
            f"experiment exited with code {returncode} without result.json: {output['stderr'][-2000:]}"
        )
    payload = read_json(result_path)
    if returncode != 0 or payload.get("status") != "ok":
        detail = payload.get("error") or output["stderr"][-2000:]
        if payload.get("traceback"):
            detail = f"{detail}\n{payload['traceback']}"
        raise ExperimentFailure(f"experiment failed: {detail}")
    return payload["result"], output


def _remaining_seconds(deadline: float) -> int:
    remaining = int(deadline - time.monotonic())
    if remaining <= 0:
        raise ExperimentFailure("generated experiment exceeded its total time budget")
    return remaining


def _stored_path(path: Path, run_root: Path | None) -> str:
    resolved = path.resolve()
    if run_root is None:
        return str(resolved)
    try:
        return resolved.relative_to(run_root.resolve()).as_posix()
    except ValueError as error:
        raise ExperimentFailure(f"artifact is outside run directory: {resolved}") from error


def _score_diagnostics(scores: np.ndarray) -> dict[str, int | float]:
    return {
        "unique_scores": int(np.unique(scores).size),
        "standard_deviation": float(np.std(scores)),
        "minimum": float(np.min(scores)),
        "maximum": float(np.max(scores)),
    }


def _projected_full_seconds(smoke_seconds: float, sample_rows: int, full_rows: int) -> float:
    """Use linear scaling as a conservative lower bound for full-data fit time."""

    if smoke_seconds <= 0 or sample_rows <= 0 or full_rows <= sample_rows:
        return max(0.0, smoke_seconds)
    startup_allowance = min(5.0, smoke_seconds)
    fit_seconds = smoke_seconds - startup_allowance
    return startup_allowance + fit_seconds * (full_rows / sample_rows)
