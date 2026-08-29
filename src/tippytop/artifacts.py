"""Atomic, append-only storage for autonomous run artifacts."""

from __future__ import annotations

import json
import hashlib
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .config import RunConfig


class HostRevisionChanged(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        # A killed run must leave either the old complete artifact or the new one.
        os.fsync(handle.fileno())
    temporary.replace(path)


def atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True, separators=(",", ":")))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def atomic_write_jsonl(path: Path, values: Iterable[Any]) -> None:
    rendered = "".join(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
        for value in values
    )
    atomic_write_text(path, rendered)


def atomic_save_npy(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.save(handle, values, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def source_revision() -> str:
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    digest = hashlib.sha256()
    root = Path(__file__).resolve().parents[2]
    paths = [
        root / "pyproject.toml",
        root / "uv.lock",
        *sorted((root / "src").rglob("*.py")),
    ]
    for path in paths:
        if not path.is_file():
            continue
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(path.read_bytes())
    # The content digest makes pre-commit runs attributable even when HEAD is unchanged.
    return f"{revision}+worktree.{digest.hexdigest()[:16]}"


def assert_source_revision(expected: str) -> None:
    current = source_revision()
    if current != expected:
        raise HostRevisionChanged(
            "host runtime changed during the run: "
            f"expected {expected}, found {current}; start a fresh run from one fixed revision"
        )


class RunStore:
    def __init__(self, path: Path, secrets: Iterable[str] = ()):
        self.path = path
        self.secrets = tuple(secret for secret in secrets if secret)

    @classmethod
    def create(cls, config: RunConfig) -> "RunStore":
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        base = config.runs_dir / f"{timestamp}-seed{config.seed}"
        path = base
        suffix = 1
        while path.exists():
            path = base.with_name(f"{base.name}-{suffix}")
            suffix += 1
        path.mkdir(parents=True)
        for directory in ("checkpoints", "experiments", "diffs", "transactions"):
            (path / directory).mkdir()
        store = cls(path, (config.api_key,))
        store.write_json("config.json", config.to_dict(redact=True))
        return store

    @classmethod
    def open(cls, path: Path, config: RunConfig) -> "RunStore":
        if not path.is_dir():
            raise ValueError(f"run directory does not exist: {path}")
        return cls(path, (config.api_key,))

    def redact(self, value: Any) -> Any:
        if isinstance(value, str):
            redacted = value
            for secret in self.secrets:
                redacted = redacted.replace(secret, "<redacted>")
            return redacted
        if isinstance(value, dict):
            return {key: self.redact(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self.redact(item) for item in value]
        if isinstance(value, tuple):
            return [self.redact(item) for item in value]
        return value

    def write_json(self, relative: str | Path, value: Any) -> None:
        atomic_write_json(self.path / relative, self.redact(value))

    def relative_path(self, path: Path) -> str:
        resolved_root = self.path.resolve()
        resolved = path.resolve()
        try:
            return resolved.relative_to(resolved_root).as_posix()
        except ValueError as error:
            raise ValueError(f"artifact is outside run directory: {resolved}") from error

    def resolve_path(self, value: str | Path) -> Path:
        path = Path(value)
        return path if path.is_absolute() else self.path / path

    def append(self, relative: str | Path, value: Any) -> None:
        append_jsonl(self.path / relative, self.redact(value))

    def event(self, event: str, **details: Any) -> None:
        self.append("events.jsonl", {"time": utc_now(), "event": event, **details})
