"""Agent run configuration."""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import sys

from .. import config as pkg


@dataclass
class AgentConfig:
    """Everything one agent run needs. Budgets follow the challenge spec."""
    data_dir: str
    run_dir: Path                       # results/runs/<run_id>
    run_id: str = "run"
    max_iters: int = 50                 # hard cap (spec)
    wall_hours: float = 6.0             # wall-clock ceiling (spec)
    iter_timeout_s: int = 900           # per-solution subprocess timeout
    temperature: float = 0.4
    conv_eps: float = pkg.CONVERGENCE_EPS   # 0.002
    conv_n: int = pkg.CONVERGENCE_N         # 3
    seed: int = pkg.DEFAULT_SEED
    final_out: Path | None = None       # where the final test submission is written
    valid_split: str = "valid"
    test_split: str = "test"
    python_exe: str = field(default_factory=lambda: sys.executable)

    @property
    def wall_budget_s(self) -> float:
        return self.wall_hours * 3600.0
