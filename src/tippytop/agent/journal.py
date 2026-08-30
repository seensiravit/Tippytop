"""Per-iteration run log (deliverable): JSONL trace + human-readable report."""
from __future__ import annotations
from dataclasses import dataclass, asdict, field
from pathlib import Path
import json

from .. import config as pkg


@dataclass
class IterationRecord:
    run_id: str
    iter: int
    phase: str                          # SEED | IMPROVE | DEBUG
    hypothesis: str
    code_path: str = ""
    diff_path: str | None = None
    exec_ok: bool = False
    timed_out: bool = False
    returncode: int | None = None
    valid_metrics: dict | None = None   # valid ONLY (never test)
    accepted: bool = False
    error_kind: str | None = None       # guard|timeout|runtime|parse|output_invalid
    error_msg: str | None = None
    recovery: str | None = None         # reprompt|debug_next
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    wall_s: float = 0.0
    cum_wall_s: float = 0.0
    cum_tokens: int = 0

    def valid_primary(self) -> float | None:
        return None if not self.valid_metrics else self.valid_metrics.get("primary")


class Journal:
    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.run_dir / "journal.jsonl"
        self.report_path = self.run_dir / "report.md"
        self.records: list[IterationRecord] = []

    def append(self, rec: IterationRecord) -> None:
        self.records.append(rec)
        with open(self.jsonl_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(rec)) + "\n")
            fh.flush()

    def write_report(self, *, run_id: str, llm_model: str, stop_reason: str,
                     best_iter: int, best_metrics: dict | None,
                     final_test_metrics: dict | None, interventions: int) -> None:
        bp = None if not best_metrics else best_metrics.get("primary")
        tp = None if not final_test_metrics else final_test_metrics.get("primary")
        lines = [
            f"# Agent run `{run_id}`", "",
            f"- LLM: `{llm_model}`",
            f"- Iterations: {len(self.records)}  |  stop reason: **{stop_reason}**",
            f"- Best valid primary: **{_f(bp)}** (iter {best_iter})",
            f"- Final **test** primary: **{_f(tp)}**  "
            f"(FM {pkg.FM_BASELINE_PRIMARY} | oracle {pkg.ORACLE_CEILING_PRIMARY}"
            + (f" | Δ vs FM {tp - pkg.FM_BASELINE_PRIMARY:+.4f}" if tp is not None else "")
            + ")",
            f"- Total tokens: {sum(r.total_tokens for r in self.records):,}",
            f"- Total wall-clock: {self.records[-1].cum_wall_s:.1f}s" if self.records else "- Total wall-clock: 0s",
            f"- Manual interventions: **{interventions}**", "",
            "## Iterations", "",
            "| # | phase | hypothesis | valid primary | accepted | error/recovery | tokens | wall(s) |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for r in self.records:
            err = r.error_kind or ""
            if r.recovery:
                err = f"{err}→{r.recovery}" if err else r.recovery
            lines.append(
                f"| {r.iter} | {r.phase} | {_short(r.hypothesis)} | "
                f"{_f(r.valid_primary())} | {'✓' if r.accepted else ''} | {err} | "
                f"{r.total_tokens} | {r.wall_s:.1f} |")
        self.report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _f(x) -> str:
    return "—" if x is None else f"{x:.4f}"


def _short(s: str, n: int = 80) -> str:
    s = " ".join((s or "").split())
    return s if len(s) <= n else s[: n - 1] + "…"
