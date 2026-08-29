"""Filesystem- and network-isolated execution for LLM-generated experiments."""

from __future__ import annotations

import os
import pickle
import resource
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from ..research.data import load_research_frames, prediction_view
from ..starter import STARTER_DIR


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
TEMP_ROOT = Path("/tmp/opencode")


class SandboxFailure(RuntimeError):
    pass


def prepare_research_data(
    path: Path,
    data_dir: Path,
    *,
    summary_path: Path | None = None,
) -> Path:
    """Persist labeled train and feature-only validation data for generated code."""

    if path.is_file():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frames = load_research_frames(data_dir)
    if summary_path is not None:
        from ..artifacts import atomic_write_json
        from ..research.data import AUXILIARY_COLUMNS, PREDICTION_COLUMNS

        train = frames["train"]
        atomic_write_json(
            summary_path,
            {
                "representation": "pandas.DataFrame",
                "prediction_columns": PREDICTION_COLUMNS,
                "training_only_columns": ["long_view", *AUXILIARY_COLUMNS],
                "training_signal_means": {
                    column: float(train[column].mean())
                    for column in ["long_view", *AUXILIARY_COLUMNS]
                },
            },
        )
    with temporary.open("wb") as handle:
        pickle.dump(
            frames,
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return path


def predict_generated(
    checkpoint_dir: Path,
    rows: pd.DataFrame,
    *,
    timeout: int = 1800,
) -> np.ndarray:
    manifest_path = checkpoint_dir / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"generated checkpoint manifest is missing: {checkpoint_dir}")
    from ..artifacts import atomic_write_json, read_json

    manifest = read_json(manifest_path)
    source_hash = manifest.get("source_hash")
    if not isinstance(source_hash, str) or not source_hash:
        raise ValueError("generated checkpoint manifest has no source hash")
    TEMP_ROOT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="tippytop-predict-", dir=TEMP_ROOT) as directory:
        workspace = Path(directory)
        rows_path = workspace / "rows.pkl"
        with rows_path.open("wb") as handle:
            # Hidden labels and contemporaneous outcomes never cross this boundary.
            pickle.dump(prediction_view(rows), handle, protocol=pickle.HIGHEST_PROTOCOL)
        request = {
            "mode": "generated_predict",
            "source_path": str((checkpoint_dir / "experiment.py").resolve()),
            "model_path": str((checkpoint_dir / "model.pkl").resolve()),
            "rows_path": str(rows_path.resolve()),
            "scores_path": str((workspace / "scores.npy").resolve()),
            "source_hash": source_hash,
        }
        request_path = workspace / "request.json"
        result_path = workspace / "result.json"
        atomic_write_json(request_path, request)
        completed = run_worker_sandboxed(
            request_path,
            result_path,
            writable_dir=workspace,
            readonly_paths=[checkpoint_dir.resolve()],
            timeout=timeout,
        )
        result = read_json(result_path)
        if completed.returncode != 0 or result.get("status") != "ok":
            detail = result.get("error") or completed.stderr[-2000:]
            raise SandboxFailure(f"generated prediction failed: {detail}")
        scores = np.load(workspace / "scores.npy", allow_pickle=False)
        if scores.shape != (len(rows),) or not np.isfinite(scores).all():
            raise SandboxFailure("generated prediction returned invalid scores")
        return np.asarray(scores, dtype=np.float32)


def run_worker_sandboxed(
    request_path: Path,
    result_path: Path,
    *,
    writable_dir: Path,
    readonly_paths: Sequence[Path],
    timeout: int,
) -> subprocess.CompletedProcess[str]:
    bubblewrap = shutil.which("bwrap")
    if bubblewrap is None:
        raise SandboxFailure("bubblewrap (bwrap) is required for generated-code execution")

    command = [
        bubblewrap,
        "--unshare-net",
        "--unshare-pid",
        "--unshare-ipc",
        "--unshare-uts",
        "--unshare-cgroup-try",
        "--die-with-parent",
        "--new-session",
        "--tmpfs",
        "/tmp",
    ]
    for system_path in (Path("/usr"), Path("/lib"), Path("/lib64")):
        if system_path.exists():
            command.extend(["--ro-bind", str(system_path), str(system_path)])

    base_prefix = Path(getattr(sys, "_base_executable", sys.executable)).resolve().parents[1]
    for runtime_path in (base_prefix, Path(sys.prefix).resolve(), PACKAGE_ROOT, STARTER_DIR):
        command.extend(["--ro-bind", str(runtime_path), str(runtime_path)])
    writable = writable_dir.resolve()
    # A private /tmp hides host paths, so recreate only ancestors needed by explicit mounts.
    for directory in _temporary_mount_directories([*readonly_paths, writable]):
        command.extend(["--dir", str(directory)])
    for path in readonly_paths:
        resolved = path.resolve()
        command.extend(["--ro-bind", str(resolved), str(resolved)])

    command.extend(
        [
            "--bind",
            str(writable),
            str(writable),
            "--proc",
            "/proc",
            "--dev",
            "/dev",
            "--chdir",
            str(writable),
            sys.executable,
            "-m",
            "tippytop.runtime.worker",
            str(request_path.resolve()),
            str(result_path.resolve()),
        ]
    )
    environment = {
        "HOME": "/tmp",
        "PATH": "/usr/bin:/bin",
        "PYTHONHASHSEED": "0",
        "PYTHONPATH": str(PACKAGE_ROOT),
        "TMPDIR": "/tmp",
        "OMP_NUM_THREADS": "4",
        "OPENBLAS_NUM_THREADS": "4",
        "MKL_NUM_THREADS": "4",
    }
    try:
        return subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            env=environment,
            preexec_fn=lambda: _set_resource_limits(timeout),
        )
    except subprocess.TimeoutExpired as error:
        raise SandboxFailure(f"generated experiment timed out after {timeout}s") from error


def _set_resource_limits(timeout: int) -> None:
    resource.setrlimit(resource.RLIMIT_CPU, (timeout, timeout + 5))
    memory_limit = 16 * 1024**3
    resource.setrlimit(resource.RLIMIT_AS, (memory_limit, memory_limit))
    resource.setrlimit(resource.RLIMIT_FSIZE, (4 * 1024**3, 4 * 1024**3))
    resource.setrlimit(resource.RLIMIT_NOFILE, (256, 256))
    # RLIMIT_NPROC is user-wide on Linux, so lowering it here can prevent bubblewrap from
    # creating its own namespace when unrelated processes already exceed the new limit.


def _temporary_mount_directories(paths: Sequence[Path]) -> list[Path]:
    directories: set[Path] = set()
    temporary_root = Path("/tmp")
    for raw_path in paths:
        path = raw_path.resolve()
        if path != temporary_root and temporary_root not in path.parents:
            continue
        directory = path if path.is_dir() else path.parent
        while directory != temporary_root:
            directories.add(directory)
            directory = directory.parent
    return sorted(directories, key=lambda value: len(value.parts))
