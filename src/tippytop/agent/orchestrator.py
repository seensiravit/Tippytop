"""The autonomous iterate loop: seed -> improve/debug -> score valid -> finalize."""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import difflib
import shutil
import time

from .config import AgentConfig
from .contract import seed_solution_source
from .convergence import ConvergenceTracker, BudgetManager
from .guard import scan as guard_scan
from .journal import Journal, IterationRecord
from .llm.base import BaseLLMClient, LLMError
from . import prompts
from .parsing import parse_response
from .sandbox import run_solution
from .scoring import score_submission
from ..data.dataset import load_dataset


@dataclass
class AgentState:
    run_id: str
    best_code: str | None = None
    best_metrics: dict | None = None        # valid ONLY
    best_iter: int = -1
    last_code: str | None = None
    last_exec_ok: bool = False
    last_error_kind: str | None = None
    last_stderr: str = ""
    interventions: int = 0
    tokens_total: int = 0
    t_start: float = field(default_factory=time.time)
    phase: str = "SEED"
    stop_reason: str = ""
    final_test_metrics: dict | None = None
    final_out: str | None = None


def run_agent(cfg: AgentConfig, llm: BaseLLMClient, *, verbose: bool = True) -> AgentState:
    journal = Journal(cfg.run_dir)
    data = load_dataset(cfg.data_dir)
    valid_rows = data.splits[cfg.valid_split]
    data_sum = prompts.data_summary(data)
    system = prompts.system_prompt()

    conv = ConvergenceTracker(cfg.conv_eps, cfg.conv_n)
    budget = BudgetManager(cfg.max_iters, cfg.wall_budget_s)
    state = AgentState(run_id=cfg.run_id)

    n = 0
    while True:
        t_iter = time.time()
        iter_dir = cfg.run_dir / f"iter_{n}"
        phase, hypothesis, code, tokens, p_tok, c_tok, error_kind, recovery = \
            _propose(cfg, llm, state, system, data_sum, journal, n)

        # guard + execute (unless proposal already failed to parse)
        exec_ok = False
        timed_out = False
        rc: int | None = None
        metrics: dict | None = None
        stderr_tail = ""
        if code and error_kind is None:
            violations = guard_scan(code)
            if violations:
                error_kind = "guard"
                stderr_tail = "; ".join(violations)
            else:
                res = run_solution(code, iter_dir=iter_dir, data_dir=cfg.data_dir,
                                   split=cfg.valid_split, timeout_s=cfg.iter_timeout_s,
                                   python_exe=cfg.python_exe)
                rc, timed_out = res.returncode, res.timed_out
                stderr_tail = res.stderr
                if res.timed_out:
                    error_kind = "timeout"
                elif not res.ok:
                    error_kind = "runtime"
                else:
                    try:
                        metrics = score_submission(res.out_path, valid_rows)
                        exec_ok = True
                    except Exception as e:              # read_submission/evaluate
                        error_kind = "output_invalid"
                        stderr_tail = f"{type(e).__name__}: {e}"

        # keep-best (greedy hill-climb on valid primary)
        prev_best_code = state.best_code or ""
        accepted = False
        if metrics is not None:
            prim = metrics["primary"]
            if state.best_metrics is None or prim > state.best_metrics["primary"] + 1e-9:
                state.best_code, state.best_metrics, state.best_iter = code, metrics, n
                accepted = True

        # persist artifacts + record
        code_path = str(iter_dir / "solution.py") if code else ""
        diff_path = _write_diff(iter_dir, prev_best_code, code) if code else None
        if error_kind is not None and n > 0:
            recovery = "debug_next"
        state.last_code = code or state.last_code
        state.last_exec_ok = exec_ok
        state.last_error_kind = error_kind
        state.last_stderr = stderr_tail
        state.tokens_total += tokens

        rec = IterationRecord(
            run_id=cfg.run_id, iter=n, phase=phase, hypothesis=hypothesis,
            code_path=code_path, diff_path=diff_path, exec_ok=exec_ok,
            timed_out=timed_out, returncode=rc, valid_metrics=metrics,
            accepted=accepted, error_kind=error_kind, error_msg=stderr_tail or None,
            recovery=recovery, prompt_tokens=p_tok, completion_tokens=c_tok,
            total_tokens=tokens, wall_s=time.time() - t_iter,
            cum_wall_s=time.time() - state.t_start, cum_tokens=state.tokens_total)
        journal.append(rec)
        if verbose:
            _print_iter(rec)

        conv.update(metrics["primary"] if metrics else None)
        n += 1

        done, why = budget.exhausted(n)
        if not done:
            cdone, cwhy = conv.converged()
            done, why = cdone, cwhy
        if done:
            state.stop_reason = why
            break

    _finalize(cfg, state, journal, llm.model, verbose)
    return state


def _propose(cfg, llm, state, system, data_sum, journal, n):
    """Return (phase, hypothesis, code, tokens, p_tok, c_tok, error_kind, recovery)."""
    if n == 0:
        return ("SEED", "Seed: reproduce the FM baseline.",
                seed_solution_source(), 0, 0, 0, None, None)

    if state.last_exec_ok:
        phase = "IMPROVE"
        user = prompts.improve_prompt(
            data_sum=data_sum, best_code=state.best_code or "",
            best_metrics=state.best_metrics,
            history=prompts.history_digest(journal.records))
    else:
        phase = "DEBUG"
        user = prompts.debug_prompt(
            data_sum=data_sum, failing_code=state.last_code or "",
            error_kind=state.last_error_kind or "unknown",
            stderr_tail=state.last_stderr,
            history=prompts.history_digest(journal.records))

    # up to two attempts: reprompt once if the reply has no code fence
    tokens = p_tok = c_tok = 0
    hypothesis, code, recovery, error_kind = "", "", None, None
    for attempt in range(2):
        try:
            resp = llm.generate(system, user, temperature=cfg.temperature)
        except LLMError as e:
            return (phase, f"(LLM error: {e})", "", tokens, p_tok, c_tok,
                    "llm_error", "debug_next")
        tokens += resp.total_tokens
        p_tok += resp.prompt_tokens
        c_tok += resp.completion_tokens
        parsed = parse_response(resp.text)
        hypothesis = parsed.hypothesis
        if parsed.parse_ok:
            return (phase, hypothesis, parsed.code, tokens, p_tok, c_tok, None, None)
        error_kind, recovery = "parse", "reprompt"
        user = user + "\n\nYour previous reply had no ```python code fence. " \
                      "Reply again with the full solution.py in a python fence."
    return (phase, hypothesis, "", tokens, p_tok, c_tok, error_kind, "debug_next")


def _finalize(cfg, state, journal, llm_model, verbose):
    """Run the valid-best solution on TEST exactly once (harness, not the LLM)."""
    if state.best_code is not None:
        state.phase = "FINALIZE"
        final_dir = cfg.run_dir / "final"
        res = run_solution(state.best_code, iter_dir=final_dir,
                           data_dir=cfg.data_dir, split=cfg.test_split,
                           timeout_s=cfg.iter_timeout_s, python_exe=cfg.python_exe)
        if res.ok:
            test_rows = load_dataset(cfg.data_dir).splits[cfg.test_split]
            try:
                state.final_test_metrics = score_submission(res.out_path, test_rows)
            except Exception:
                state.final_test_metrics = None
            out = Path(cfg.final_out) if cfg.final_out else (cfg.run_dir / "agent_test.csv")
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(res.out_path, out)
            state.final_out = str(out)
    state.phase = "DONE"
    journal.write_report(
        run_id=cfg.run_id, llm_model=llm_model, stop_reason=state.stop_reason,
        best_iter=state.best_iter, best_metrics=state.best_metrics,
        final_test_metrics=state.final_test_metrics, interventions=state.interventions)
    if verbose:
        print(f"\n[agent] stop={state.stop_reason} best valid "
              f"{_f(state.best_metrics)} @iter {state.best_iter} | "
              f"final test {_f(state.final_test_metrics)} | report {journal.report_path}")


# --- small helpers -------------------------------------------------------

def _write_diff(iter_dir: Path, base: str, code: str) -> str:
    iter_dir.mkdir(parents=True, exist_ok=True)
    diff = difflib.unified_diff((base or "").splitlines(True),
                                (code or "").splitlines(True),
                                fromfile="best", tofile="proposed")
    p = iter_dir / "diff.patch"
    p.write_text("".join(diff), encoding="utf-8")
    return str(p)


def _print_iter(r: IterationRecord) -> None:
    vp = "—" if not r.valid_metrics else f"{r.valid_metrics['primary']:.4f}"
    tag = "" if r.error_kind is None else f" [{r.error_kind}]"
    star = " *best*" if r.accepted else ""
    print(f"[iter {r.iter}] {r.phase:7s} valid {vp}{tag}{star} "
          f"| tok {r.total_tokens} | {r.wall_s:.1f}s")


def _f(m: dict | None) -> str:
    return "—" if not m else f"{m.get('primary'):.4f}"
