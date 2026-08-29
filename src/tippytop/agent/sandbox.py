"""Run a generated solution.py in an isolated subprocess with a wall-clock timeout.

Best-effort isolation: own working dir per iteration, captured stdout/stderr, hard
timeout with process-tree kill. Not a security sandbox (see guard.py). The child
env has GEMINI_API_KEY stripped so generated code can never read the API key.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import os
import subprocess
import sys
import time

_TAIL = 6000  # bytes of stdout/stderr kept for prompts


@dataclass
class ExecResult:
    ok: bool
    returncode: int | None
    stdout: str
    stderr: str
    timed_out: bool
    wall_s: float
    out_path: Path | None


def _kill_tree(proc: subprocess.Popen) -> None:
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                           capture_output=True)
        else:
            proc.kill()
    except Exception:
        pass
    try:
        proc.kill()
    except Exception:
        pass


def run_solution(code: str, *, iter_dir: Path, data_dir, split: str,
                 timeout_s: int, python_exe: str = sys.executable) -> ExecResult:
    iter_dir = Path(iter_dir)
    iter_dir.mkdir(parents=True, exist_ok=True)
    sol_path = iter_dir / "solution.py"
    out_path = iter_dir / "scores.csv"
    sol_path.write_text(code, encoding="utf-8")

    env = dict(os.environ)
    env.pop("GEMINI_API_KEY", None)          # never expose the key to generated code
    env["PYTHONHASHSEED"] = "0"

    creationflags = 0
    if os.name == "nt":
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]

    cmd = [python_exe, str(sol_path), "--data_dir", str(data_dir),
           "--split", split, "--out", str(out_path)]

    t0 = time.time()
    proc = subprocess.Popen(cmd, cwd=str(iter_dir), stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True, env=env,
                            creationflags=creationflags)
    timed_out = False
    try:
        out, err = proc.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        timed_out = True
        _kill_tree(proc)
        try:
            out, err = proc.communicate(timeout=15)
        except subprocess.TimeoutExpired:
            out, err = "", "process did not terminate after kill"
    wall_s = time.time() - t0

    (iter_dir / "stdout.txt").write_text(out or "", encoding="utf-8")
    (iter_dir / "stderr.txt").write_text(err or "", encoding="utf-8")

    rc = proc.returncode
    produced = out_path if out_path.exists() else None
    ok = (not timed_out) and rc == 0 and produced is not None
    if timed_out:
        err = (err or "") + f"\n[timeout after {timeout_s}s]"
    return ExecResult(ok=ok, returncode=rc, stdout=_tail(out), stderr=_tail(err),
                      timed_out=timed_out, wall_s=wall_s, out_path=produced)


def _tail(s: str | None) -> str:
    s = s or ""
    return s if len(s) <= _TAIL else "...(truncated)...\n" + s[-_TAIL:]
