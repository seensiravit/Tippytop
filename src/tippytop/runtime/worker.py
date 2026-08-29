"""Worker entry point for trusted baselines and sandboxed generated code."""

from __future__ import annotations

import importlib.util
import json
import pickle
import random
import sys
import time
import traceback
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

from ..artifacts import atomic_write_json, read_json
from ..experiments import train_parametric
from ..generated import GeneratedExperiment
from ..research.data import prediction_view
from ..starter import load_splits


REFERENCE_PARAMETERS: dict[str, int | float] = {
    "learning_rate": 0.001,
    "epochs": 40,
    "l2": 1e-6,
    "batch_size": 8192,
    "patience": 4,
    "seed": 0,
    "embedding_dim": 16,
}
GENERATED_MODULE = "_tippytop_generated_experiment"


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: python -m tippytop.runtime.worker REQUEST_JSON RESULT_JSON", file=sys.stderr)
        return 2
    request_path, result_path = map(Path, sys.argv[1:])
    try:
        request = read_json(request_path)
        mode = request["mode"]
        if mode == "reference_baseline":
            result = _run_reference_baseline(request)
        elif mode == "generated_smoke":
            result = _run_generated_smoke(request)
        elif mode == "generated_train":
            result = _run_generated_train(request)
        elif mode == "generated_predict":
            result = _run_generated_predict(request)
        else:
            raise ValueError(f"unsupported worker mode: {mode!r}")
        atomic_write_json(result_path, {"status": "ok", "result": result})
        return 0
    except BaseException as error:  # SystemExit from generated code must not forge a successful run.
        atomic_write_json(
            result_path,
            {
                "status": "error",
                "error": str(error),
                "traceback": traceback.format_exc(),
            },
        )
        traceback.print_exc()
        return 1


def _run_reference_baseline(request: dict[str, Any]) -> dict[str, Any]:
    parameters = dict(REFERENCE_PARAMETERS)
    parameters["seed"] = int(request["seed"])
    splits = load_splits(request["data_dir"])
    # The reference trainer needs no test rows while selecting its validation checkpoint.
    research_splits = {name: splits[name] for name in ("train", "valid")}
    return train_parametric(
        research_splits,
        Path(request["artifact_dir"]),
        model_type="fm",
        objective="pointwise",
        parameters=parameters,
        verbose=True,
    )


def _run_generated_train(request: dict[str, Any]) -> dict[str, Any]:
    started = time.monotonic()
    with Path(request["data_path"]).open("rb") as handle:
        splits = pickle.load(handle)
    train_rows = splits["train"]
    valid_rows = splits["valid"]
    module = _load_generated_module(Path(request["source_path"]), request["source_hash"])

    seed = int(request["seed"])
    _seed_generated_runtime(seed)
    model, metadata = _split_model_metadata(module.fit(train_rows, seed))
    scores = _validated_ranking_scores(
        module.predict(model, prediction_view(valid_rows)),
        len(valid_rows),
    )

    artifact_dir = Path(request["artifact_dir"])
    model_path = artifact_dir / "model.pkl"
    with model_path.open("wb") as handle:
        pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
    np.save(artifact_dir / "valid_scores.npy", scores, allow_pickle=False)
    safe_metadata = _json_safe(metadata)
    duration = time.monotonic() - started
    print(f"generated rows={len(valid_rows)} duration={duration:.1f}s", flush=True)
    return {
        "model_type": "generated",
        "checkpoint": str(artifact_dir),
        "valid_scores": str(artifact_dir / "valid_scores.npy"),
        "history": safe_metadata.get("history", []),
        "epochs_completed": int(safe_metadata.get("epochs_completed", 0)),
        "pairs": int(safe_metadata.get("pairs", 0)),
        "duration_seconds": duration,
        "metadata": safe_metadata,
    }


def _run_generated_smoke(request: dict[str, Any]) -> dict[str, Any]:
    with Path(request["data_path"]).open("rb") as handle:
        splits = pickle.load(handle)
    rng = np.random.default_rng(int(request["seed"]))
    train_rows = splits["train"]
    sample_size = min(10_000, len(train_rows))
    indices = np.sort(rng.choice(len(train_rows), size=sample_size, replace=False))
    sample = train_rows.iloc[indices].reset_index(drop=True)
    prediction_rows = prediction_view(splits["valid"].iloc[: min(512, len(splits["valid"]))])
    module = _load_generated_module(Path(request["source_path"]), request["source_hash"])

    seed = int(request["seed"])
    _seed_generated_runtime(seed)
    model, _ = _split_model_metadata(module.fit(sample, seed))
    _validated_ranking_scores(module.predict(model, prediction_rows), len(prediction_rows))
    print(f"generated smoke rows={sample_size} predictions={len(prediction_rows)}", flush=True)
    return {"train_rows": sample_size, "prediction_rows": len(prediction_rows)}


def _run_generated_predict(request: dict[str, Any]) -> dict[str, Any]:
    source_path = Path(request["source_path"])
    # Replay only the exact source selected and recorded with this checkpoint.
    module = _load_generated_module(source_path, request["source_hash"])
    with Path(request["model_path"]).open("rb") as handle:
        model = pickle.load(handle)
    with Path(request["rows_path"]).open("rb") as handle:
        rows = pickle.load(handle)
    scores = _validated_ranking_scores(module.predict(model, rows), len(rows))
    np.save(request["scores_path"], scores, allow_pickle=False)
    return {"rows": len(rows)}


def _load_generated_module(source_path: Path, expected_hash: str) -> ModuleType:
    source = source_path.read_text(encoding="utf-8")
    experiment = GeneratedExperiment.from_dict(
        {"hypothesis": "sandbox execution", "expected_effect": "measured validation", "source": source}
    )
    if experiment.source_hash != expected_hash:
        raise ValueError("generated source hash does not match the request")
    spec = importlib.util.spec_from_file_location(GENERATED_MODULE, source_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load generated experiment module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[GENERATED_MODULE] = module
    spec.loader.exec_module(module)
    return module


def _validated_scores(scores: Any, expected_rows: int) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float32)
    if values.shape != (expected_rows,):
        raise ValueError(f"predict returned shape {values.shape}; expected ({expected_rows},)")
    if not np.isfinite(values).all():
        raise ValueError("predict returned NaN or infinite scores")
    return values


def _validated_ranking_scores(scores: Any, expected_rows: int) -> np.ndarray:
    values = _validated_scores(scores, expected_rows)
    if len(values) >= 100 and np.unique(values).size <= 2:
        raise ValueError(
            "predict returned class labels rather than continuous ranking scores; use "
            "predict_proba, decision_function, regression output, or model logits"
        )
    return values


def _split_model_metadata(fitted: Any) -> tuple[Any, dict[str, Any]]:
    if isinstance(fitted, tuple) and len(fitted) == 2 and isinstance(fitted[1], dict):
        return fitted
    # Tuples are valid model bundles, for example (encoder, sklearn_estimator).
    return fitted, {}


def _seed_generated_runtime(seed: int) -> None:
    """Make common global RNG use reproducible even when generated code omits local seeding."""

    random.seed(seed)
    np.random.seed(seed)


def _json_safe(value: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(value, default=str))


if __name__ == "__main__":
    raise SystemExit(main())
