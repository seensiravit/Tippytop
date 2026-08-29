"""Label-safe benchmark summaries and LLM research context."""

from __future__ import annotations

import collections
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .artifacts import RunStore, read_json, read_jsonl
from .generated import parse_json_object
from .starter import evaluate


PUBLIC_PROBLEM_INSIGHTS = [
    "The official FM is trained with pointwise binary cross-entropy although evaluation is within-user ranking.",
    "Pairwise or listwise ranking losses are the highest-priority unmeasured objective direction.",
    "Organizer measurements found static side features and larger embedding dimensions flat within noise.",
    "Features constant within one user's impressions cannot change that user's ranking unless crossed with item-side signals.",
    "Training exposes watch time and auxiliary engagement outcomes for auxiliary supervision or historical aggregates; prediction exposes only impression-time context, including hourmin and time_ms.",
    "Outcome-derived training histories must be strictly past-only in time_ms for each row; prediction may use mappings fitted on the complete training period.",
    "TabularEncoder intentionally covers only starter-compatible fields, so temporal, auxiliary, recency, or custom crossed features require direct pandas feature construction.",
    "A feature-engineering hypothesis is implemented only if those values reach the final model input in both fit and predict; merely adding ignored DataFrame columns is a no-op.",
    "For pairwise learning, pointwise initialization and same-user hard negatives near the current top scores target rank errors more directly than one fixed set of uniform negatives.",
]

RECENT_OUTCOME_LIMIT = 6
RECENT_CODE_LIMIT = 2
SOURCE_CONTEXT_CHARS = 7_000
ERROR_CONTEXT_CHARS = 2_000


def random_sanity(splits: dict[str, list[tuple[Any, ...]]]) -> dict[str, dict[str, float]]:
    values: dict[str, list[dict[str, Any]]] = {"valid": [], "test": []}
    for seed in range(5):
        rng = np.random.default_rng(seed)
        for split in ("valid", "test"):
            rows = splits[split]
            values[split].append(
                evaluate(
                    [row[1] for row in rows],
                    [row[6] for row in rows],
                    rng.random(len(rows)),
                )
            )
    return {
        split: {
            metric: float(np.mean([value[metric] for value in metrics]))
            for metric in ("GAUC", "nDCG@5", "primary")
        }
        for split, metrics in values.items()
    }


def summarize_dataset(splits: dict[str, list[tuple[Any, ...]]]) -> dict[str, Any]:
    """Summarize only train and public validation data for LLM consumption."""

    summary: dict[str, Any] = {}
    for split in ("train", "valid"):
        rows = splits[split]
        by_user: dict[str, list[int]] = collections.defaultdict(list)
        items: set[str] = set()
        for row in rows:
            by_user[str(row[1])].append(int(row[6]))
            items.add(str(row[2]))
        impressions = np.asarray([len(values) for values in by_user.values()], dtype=np.int64)
        positives = np.asarray([sum(values) for values in by_user.values()], dtype=np.int64)
        summary[split] = {
            "rows": len(rows),
            "users": len(by_user),
            "items": len(items),
            "positive_rate": float(sum(int(row[6]) for row in rows) / max(1, len(rows))),
            "impressions_per_user": _distribution(impressions),
            "positives_per_user": _distribution(positives),
            "all_negative_users": int(np.sum(positives == 0)),
            "all_positive_users": int(np.sum(positives == impressions)),
            "discriminative_users": int(np.sum((positives > 0) & (positives < impressions))),
            "date_min": min((int(row[0]) for row in rows), default=None),
            "date_max": max((int(row[0]) for row in rows), default=None),
        }
    return summary


def build_research_context(
    store: RunStore,
    state: dict[str, Any],
    history: Sequence[dict[str, Any]],
    best_source: str,
) -> dict[str, Any]:
    prior_path = store.path / "prior_research.json"
    prior = read_json(prior_path) if prior_path.is_file() else {}
    recent_experiments = [
        *(prior.get("recent_experiments") or []),
        *_outcome_summaries(history),
    ][-RECENT_OUTCOME_LIMIT:]
    recent_code_attempts = [
        *(prior.get("recent_code_attempts") or []),
        *_recent_code_attempts(store, history),
    ][-RECENT_CODE_LIMIT:]
    return {
        "task": "within-user impression ranking of long_view; maximize mean(GAUC, nDCG@5)",
        "public_problem_insights": PUBLIC_PROBLEM_INSIGHTS,
        "constraints": {
            "validation_only": True,
            "epsilon": state["config"]["epsilon"],
            "target_primary": float(state["baseline_valid"]["primary"])
            + float(state["config"]["epsilon"]),
            "remaining_iterations": state["config"]["max_iterations"] - state["iteration"],
            "experiment_timeout_seconds": state["config"]["experiment_timeout"],
        },
        "dataset_summary": read_json(store.path / "dataset_summary.json"),
        "research_data_schema": (
            read_json(store.path / "research_schema.json")
            if (store.path / "research_schema.json").is_file()
            else None
        ),
        "baseline_validation": state["baseline_valid"],
        "current_best": state["best"]["metrics"],
        "current_best_source": best_source,
        "prior_research_note": prior.get("context_note"),
        "recent_experiments": recent_experiments,
        "recent_code_attempts": recent_code_attempts,
    }


def snapshot_prior_research(store: RunStore, prior_run: Path) -> dict[str, Any]:
    prior_path = prior_run.resolve()
    prior_state = read_json(prior_path / "state.json")
    if prior_state.get("test_evaluated") is not False:
        raise ValueError("prior research run must not have evaluated test")
    iteration_history = read_jsonl(prior_path / "iterations.jsonl")
    prior_store = RunStore(prior_path)
    payload = {
        "source_run": str(prior_path),
        "context_note": (
            "Prior outcomes are research memory, not active-run state; remeasure promising code "
            "because host implementation changes may affect old metrics."
        ),
        "recent_experiments": _outcome_summaries(iteration_history),
        "recent_code_attempts": _recent_code_attempts(prior_store, iteration_history),
        "used_source_hashes": list(prior_state.get("used_source_hashes", [])),
    }
    store.write_json("prior_research.json", payload)
    return payload


def _outcome_summaries(history: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "iteration": record.get("iteration"),
            "hypothesis": record.get("hypothesis"),
            "source_hash": record.get("source_hash"),
            "metrics": record.get("metrics"),
            "diagnostics": _context_diagnostics(record),
            "status": record.get("status"),
            "error": _bounded_text(str(record.get("error") or ""), ERROR_CONTEXT_CHARS),
            "reflection": _bounded_text(
                str(record.get("reflection") or ""),
                ERROR_CONTEXT_CHARS,
            ),
        }
        for record in history[-RECENT_OUTCOME_LIMIT:]
    ]


def _recent_code_attempts(
    store: RunStore,
    history: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    attempts: list[dict[str, Any]] = []
    for record in reversed(history):
        if len(attempts) >= RECENT_CODE_LIMIT:
            break
        iteration = record.get("iteration")
        if not isinstance(iteration, int):
            continue
        artifact_path = store.path / "experiments" / f"{iteration:03d}.json"
        artifact = read_json(artifact_path) if artifact_path.is_file() else {}
        experiment = _recorded_experiment(artifact, record)
        source = experiment.get("source")
        if not isinstance(source, str):
            continue
        recovery = artifact.get("recovery") or record.get("recovery") or []
        attempts.append(
            {
                "iteration": iteration,
                "status": record.get("status"),
                "hypothesis": experiment.get("hypothesis") or record.get("hypothesis"),
                "expected_effect": experiment.get("expected_effect") or record.get("expected_effect"),
                "source_hash": artifact.get("source_hash") or record.get("source_hash"),
                "source": _bounded_text(source, SOURCE_CONTEXT_CHARS),
                "metrics": record.get("metrics"),
                "diagnostics": _context_diagnostics(record),
                "error": _bounded_text(str(record.get("error") or ""), ERROR_CONTEXT_CHARS),
                "recovery": [
                    {
                        key: (
                            _bounded_text(str(item[key]), ERROR_CONTEXT_CHARS)
                            if key == "error" and item.get(key)
                            else item.get(key)
                        )
                        for key in ("action", "experiment_id", "source_hash", "error")
                        if item.get(key) is not None
                    }
                    for item in recovery[-3:]
                    if isinstance(item, dict)
                ],
            }
        )
    attempts.reverse()
    return attempts


def _recorded_experiment(
    artifact: dict[str, Any],
    record: dict[str, Any],
) -> dict[str, Any]:
    persisted = (
        artifact.get("executed_experiment")
        or artifact.get("experiment")
        or artifact.get("initial_experiment")
    )
    if isinstance(persisted, dict):
        return persisted
    for response in reversed(record.get("responses") or []):
        content = response.get("content") if isinstance(response, dict) else None
        if not isinstance(content, str):
            continue
        try:
            payload = parse_json_object(content)
        except ValueError:
            continue
        if isinstance(payload.get("source"), str):
            return payload
    return {}


def _context_diagnostics(record: dict[str, Any]) -> dict[str, Any] | None:
    diagnostics = record.get("diagnostics")
    if not isinstance(diagnostics, dict):
        return None
    rendered = dict(diagnostics)
    matching = diagnostics.get("matching_prior_iteration")
    if matching is not None:
        rendered["outcome_classification"] = "exact_repeat_likely_no_op"
    return rendered


def _bounded_text(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    marker = "\n... <truncated for context> ...\n"
    prefix = (limit - len(marker)) // 2
    suffix = limit - len(marker) - prefix
    return value[:prefix] + marker + value[-suffix:]


def _distribution(values: np.ndarray) -> dict[str, float]:
    if len(values) == 0:
        return {"min": 0.0, "median": 0.0, "p90": 0.0, "max": 0.0}
    return {
        "min": float(np.min(values)),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.9)),
        "max": float(np.max(values)),
    }
