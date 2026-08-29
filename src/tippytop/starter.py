"""Stable adapter around the organizer-provided starter kit."""

from __future__ import annotations

import importlib.util
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Any, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
STARTER_DIR = REPOSITORY_ROOT / "kuairand-starter-kit" / "kuairand-starter-kit"


def _load_module(name: str, filename: str) -> ModuleType:
    path = STARTER_DIR / filename
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load starter module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=1)
def data_module() -> ModuleType:
    return _load_module("_tippytop_starter_data", "data.py")


@lru_cache(maxsize=1)
def evaluate_module() -> ModuleType:
    return _load_module("_tippytop_starter_evaluate", "evaluate.py")


def load_splits(data_dir: str | Path) -> dict[str, list[tuple[Any, ...]]]:
    return data_module().load(str(data_dir))


def encode_splits(
    splits: dict[str, list[tuple[Any, ...]]],
) -> tuple[dict[str, tuple[Any, Any, list[str]]], int]:
    return data_module().encode(splits)


def evaluate(user_ids: Sequence[Any], labels: Sequence[Any], scores: Sequence[Any]) -> dict[str, Any]:
    return evaluate_module().evaluate(user_ids, labels, scores)
