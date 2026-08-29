"""Runtime configuration with environment and JSON serialization support."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


DEFAULT_BASE_URL = "http://100.100.10.10:12345/v1"
DEFAULT_MODEL = "gemma4:e4b"


@dataclass(frozen=True)
class RunConfig:
    data_dir: Path = Path("KuaiRand-Pure/data")
    runs_dir: Path = Path("runs")
    prior_run: Path | None = None
    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    api_key: str = ""
    max_iterations: int = 50
    max_hours: float = 6.0
    epsilon: float = 0.002
    patience: int = 3
    seed: int = 0
    experiment_timeout: int = 1800
    llm_timeout: int = 600
    offline: bool = False

    @classmethod
    def from_env(cls, **overrides: Any) -> "RunConfig":
        values: dict[str, Any] = {
            "data_dir": Path(os.getenv("TIPPYTOP_DATA_DIR", "KuaiRand-Pure/data")),
            "runs_dir": Path(os.getenv("TIPPYTOP_RUNS_DIR", "runs")),
            "base_url": os.getenv("TIPPYTOP_LLM_BASE_URL", DEFAULT_BASE_URL),
            "model": os.getenv("TIPPYTOP_LLM_MODEL", DEFAULT_MODEL),
            "api_key": os.getenv("TIPPYTOP_LLM_API_KEY", ""),
        }
        values.update({key: value for key, value in overrides.items() if value is not None})
        values["data_dir"] = Path(values["data_dir"])
        values["runs_dir"] = Path(values["runs_dir"])
        if values.get("prior_run") is not None:
            values["prior_run"] = Path(values["prior_run"])
        return cls(**values)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "RunConfig":
        restored = dict(value)
        restored["data_dir"] = Path(restored["data_dir"])
        restored["runs_dir"] = Path(restored.get("runs_dir", "runs"))
        if restored.get("prior_run") is not None:
            restored["prior_run"] = Path(restored["prior_run"])
        persisted_key = restored.get("api_key", "")
        if persisted_key == "<redacted>":
            # Secrets are never persisted; resume reacquires the key from the environment.
            restored["api_key"] = os.getenv("TIPPYTOP_LLM_API_KEY", "")
        return cls(**restored)

    def to_dict(self, *, redact: bool = True) -> dict[str, Any]:
        value = asdict(self)
        value["data_dir"] = str(self.data_dir)
        value["runs_dir"] = str(self.runs_dir)
        value["prior_run"] = str(self.prior_run) if self.prior_run is not None else None
        if redact and self.api_key:
            value["api_key"] = "<redacted>"
        return value

    def validate(self) -> None:
        if not self.base_url.startswith(("http://", "https://")):
            raise ValueError("base_url must start with http:// or https://")
        if not self.model.strip():
            raise ValueError("model cannot be empty")
        if not 1 <= self.max_iterations <= 50:
            raise ValueError("max_iterations must be between 1 and 50")
        if not 0 < self.max_hours <= 6:
            raise ValueError("max_hours must be in (0, 6]")
        if not 0 <= self.epsilon <= 0.1:
            raise ValueError("epsilon must be between 0 and 0.1")
        if not 1 <= self.patience <= 50:
            raise ValueError("patience must be between 1 and 50")
        if self.experiment_timeout <= 0 or self.llm_timeout <= 0:
            raise ValueError("timeouts must be positive")
        if self.prior_run is not None and not (self.prior_run / "state.json").is_file():
            raise ValueError(f"prior run is missing state.json: {self.prior_run}")
