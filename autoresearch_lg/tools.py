"""Non-LLM building blocks: git, subprocess, results.tsv, dashboard.

Kept separate from graph.py so the git/subprocess/logging logic can be
unit-tested (or reused by a plain script) without pulling in the Anthropic
client or LangGraph.
"""
from __future__ import annotations

import difflib
import json
import math
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import time
from pathlib import Path, PurePosixPath

# The interpreter running this harness. NOT a bare "python3" string: on Windows
# that resolves to the Microsoft Store alias stub ("Python was not found"), and
# on any platform it can pick an interpreter outside our venv (missing numpy).
PYTHON = sys.executable


def _child_env() -> dict:
    """Environment for every subprocess: force UTF-8 stdio.

    The kit's submit.py prints a U+2713 check mark on success. On Windows the
    child's stdout defaults to the ANSI code page (cp1252), which cannot encode
    it, so submit.py --check dies with UnicodeEncodeError *after* writing a
    perfectly valid submission -- and finalize reports the submission as FAILED.
    Observed on run2. PYTHONIOENCODING fixes it without touching the frozen kit.
    """
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    return env

RUN_TIMEOUT_SECONDS = 600  # program.md: kill and discard past 10 minutes
EDA_TIMEOUT_SECONDS = 60

_EDA_SCRIPT = r"""
import json, sys, collections
import numpy as np
from data import load, FIELDS

splits = load(sys.argv[1])

def stats(name):
    rows = splits[name]
    n = len(rows)
    if n == 0:
        return {"n_rows": 0}
    labels = np.array([x[6] for x in rows], dtype=np.float32)
    imp, pos = collections.Counter(), collections.Counter()
    for x in rows:
        imp[x[1]] += 1
        pos[x[1]] += x[6]
    all_neg = sum(1 for u in imp if pos[u] == 0)
    all_pos = sum(1 for u in imp if pos[u] == imp[u])
    counts = np.array(list(imp.values()))
    return {
        "n_rows": n,
        "n_users": len(imp),
        "n_videos": len(set(x[2] for x in rows)),
        "positive_rate": round(float(labels.mean()), 4),
        "users_all_negative_pct": round(100 * all_neg / len(imp), 1),
        "users_all_positive_pct": round(100 * all_pos / len(imp), 1),
        "impressions_per_user_median": float(np.median(counts)),
        "impressions_per_user_p90": float(np.percentile(counts, 90)),
    }

out = {s: stats(s) for s in ("train", "valid")}
tr = splits["train"]
out["train_cardinality"] = {
    "user_id": len(set(x[1] for x in tr)),
    "video_id": len(set(x[2] for x in tr)),
    "author_id": len(set(x[3] for x in tr)),
    "tab": len(set(x[4] for x in tr)),
}
print(json.dumps(out))
"""
# The negative lookahead matters: `[\d.]+` alone matches the "1" of "1e9" and
# silently reports a primary of 1.0. Anything that is not a plain decimal is not
# a score this harness will accept.
# Named groups because the deliverable is per-metric: the judging formula is
# mean(delta(GAUC), delta(nDCG@5)), and the results table must report both, not
# just the primary they average to.
SUMMARY_RE = re.compile(
    r"valid\s+GAUC\s+(?P<valid_GAUC>[\d.]+)(?![\w.])\s*\|\s*"
    r"nDCG@5\s+(?P<valid_nDCG>[\d.]+)(?![\w.])\s*\|\s*"
    r"primary\s+(?P<valid_primary>[\d.]+)(?![\w.]).*?"
    r"test\s+GAUC\s+(?P<test_GAUC>[\d.]+)(?![\w.])\s*\|\s*"
    r"nDCG@5\s+(?P<test_nDCG>[\d.]+)(?![\w.])\s*\|\s*"
    r"primary\s+(?P<test_primary>[\d.]+)(?![\w.])",
    re.DOTALL,
)

# mean(GAUC, nDCG@5) is a mean of two quantities that each live in [0, 1], so a
# primary outside that range is not a good result -- it is a broken or dishonest
# one. Without this bound, code that prints `primary 99.0` becomes the incumbent
# forever: `best_valid_primary` only ever moves up, propose() then bases every
# later proposal on that folder, and finalize ships it.
METRIC_MIN, METRIC_MAX = 0.0, 1.0


# ---------------------------------------------------------------- git -----
def git(repo_root: str, *args: str, check: bool = True) -> str:
    res = subprocess.run(
        ["git", *args], cwd=repo_root, capture_output=True, text=True
    )
    if check and res.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed:\n{res.stderr}")
    return res.stdout.strip()


def current_branch(repo_root: str) -> str:
    return git(repo_root, "rev-parse", "--abbrev-ref", "HEAD")


def short_head(repo_root: str) -> str:
    return git(repo_root, "rev-parse", "--short", "HEAD")


def branch_exists(repo_root: str, branch: str) -> bool:
    res = subprocess.run(
        ["git", "rev-parse", "--verify", branch],
        cwd=repo_root, capture_output=True, text=True,
    )
    return res.returncode == 0


def create_experiment_branch(repo_root: str, tag: str) -> str:
    branch = f"autoresearch/{tag}"
    if branch_exists(repo_root, branch):
        raise RuntimeError(
            f"branch {branch} already exists — pick a fresh tag "
            "(program.md requires a fresh branch per run)"
        )
    git(repo_root, "checkout", "-b", branch)
    return branch


# ------------------------------------------------------------ subprocess --
def run_baseline(cwd: str, data_dir: str, seed: int = 0) -> dict:
    """Run `<python> baseline.py --model fm` from `cwd` and capture output.

    `cwd` is whichever directory holds the baseline.py to run — normally an
    experiment folder under runs/ (see make_experiment_dir), never repo_root
    itself once an experiment has actually started (the root baseline.py is
    only ever read, never executed in-place, by this harness).

    Returns dict(stdout, wall_seconds, crashed, timed_out).
    """
    cmd = [
        PYTHON, "baseline.py",
        "--model", "fm",
        "--data_dir", data_dir,
        "--seed", str(seed),
    ]
    start = time.time()
    try:
        res = subprocess.run(
            cmd, cwd=cwd, capture_output=True, text=True, env=_child_env(),
            timeout=RUN_TIMEOUT_SECONDS,
        )
        wall = time.time() - start
        stdout = res.stdout + "\n" + res.stderr
        return {
            "stdout": stdout,
            "wall_seconds": wall,
            "crashed": res.returncode != 0,
            "timed_out": False,
        }
    except subprocess.TimeoutExpired as e:
        wall = time.time() - start
        stdout = (e.stdout or "") + "\n" + (e.stderr or "")
        return {
            "stdout": stdout,
            "wall_seconds": wall,
            "crashed": True,
            "timed_out": True,
        }


def run_eda(repo_root: str, data_dir: str) -> dict:
    """Deterministic EDA over the current train/valid splits — no LLM call.

    Grounds hypothesis proposals in real numbers (class balance, per-user
    impression counts, field cardinalities) instead of the model guessing.
    Runs against the repo's own data.py via subprocess (reflects whatever
    is on disk right now, and doesn't require numpy in the orchestrator's
    own venv).
    """
    try:
        res = subprocess.run(
            [PYTHON, "-c", _EDA_SCRIPT, data_dir],
            cwd=repo_root, capture_output=True, text=True, env=_child_env(),
            timeout=EDA_TIMEOUT_SECONDS,
        )
        if res.returncode != 0:
            return {"error": res.stderr[-2000:]}
        return json.loads(res.stdout.strip().splitlines()[-1])
    except (subprocess.TimeoutExpired, json.JSONDecodeError, IndexError) as e:
        return {"error": str(e)}


def parse_summary(stdout: str) -> tuple[float, float] | None:
    """Extract (valid_primary, test_primary) from baseline.py's summary block.

    Returns None when the block is absent OR when the numbers in it are not
    possible scores. Both cases are handled identically upstream (a clean
    `collect_metrics` failure, which routes to repair), and both mean the same
    thing: this run produced nothing we can believe.
    """
    full = parse_summary_full(stdout)
    if full is None:
        return None
    return full["valid"]["primary"], full["test"]["primary"]


def parse_summary_full(stdout: str) -> dict | None:
    """Every metric in the summary block, per split, or None if any is impossible.

    Deliverable 4 asks for validation-best GAUC and nDCG@5 and the absolute
    delta over the official baseline, so the harness has to record both metrics
    rather than only the primary. (Worth knowing: because primary is defined as
    mean(GAUC, nDCG@5), the judging formula mean(delta(GAUC), delta(nDCG@5)) is
    algebraically identical to delta(primary) -- but the table still has to show
    the two metrics.)
    """
    m = SUMMARY_RE.search(stdout)
    if not m:
        return None
    try:
        vals = {k: float(v) for k, v in m.groupdict().items()}
    except (TypeError, ValueError):
        return None
    for v in vals.values():
        if not math.isfinite(v) or not (METRIC_MIN <= v <= METRIC_MAX):
            return None
    return {
        "valid": {"GAUC": vals["valid_GAUC"], "nDCG@5": vals["valid_nDCG"],
                  "primary": vals["valid_primary"]},
        "test": {"GAUC": vals["test_GAUC"], "nDCG@5": vals["test_nDCG"],
                 "primary": vals["test_primary"]},
    }


# ------------------------------------------------------------ results.tsv-
RESULTS_HEADER = "commit\tvalid_primary\ttest_primary\twall_seconds\tstatus\tdescription\n"


# interventions.jsonl and resource_report.json belong to a single run exactly as
# runs.jsonl does. Leaving interventions.jsonl behind would carry the previous
# run's manual-intervention count into a fresh one and inflate the number that
# Deliverable 3 asks for -- the same class of bug archiving was added to fix.
RUN_ARTIFACTS = ("runs.jsonl", "results.tsv", "concepts.json", "checkpoints.db",
                 ".autoresearch_start_time", "results_dashboard.html",
                 "interventions.jsonl", "resource_report.json")


def archive_run_artifacts(repo_root: str) -> str | None:
    """Move a previous run's artifacts aside so a new tag starts from zero.

    These live at the repo root and are NOT per-tag, while bootstrap
    reconstructs `iteration` from the length of runs.jsonl. Without this, a
    fresh `setup --tag X` inherits the previous run's iteration count -- observed
    on run2, which began "3/3 iterations used", ran one experiment and stopped.

    Moved, never deleted: runs.jsonl is a required deliverable, and a run that
    is superseded is not a run that should be destroyed.

    Returns the archive directory, or None if there was nothing to move.
    """
    root = Path(repo_root)
    present = [f for f in RUN_ARTIFACTS if (root / f).exists()]
    if not present:
        return None
    dest = root / "runs" / f"_archive_{time.strftime('%Y%m%dT%H%M%S')}"
    dest.mkdir(parents=True, exist_ok=True)
    for f in present:
        shutil.move(str(root / f), str(dest / f))
    return str(dest)


def init_results_tsv(path: str) -> None:
    p = Path(path)
    if not p.exists():
        p.write_text(RESULTS_HEADER, encoding="utf-8")


def append_result(
    path: str, commit: str, valid_primary: float, test_primary: float,
    wall_seconds: float, status: str, description: str,
) -> None:
    line = (
        f"{commit}\t{valid_primary:.6f}\t{test_primary:.6f}\t"
        f"{wall_seconds:.1f}\t{status}\t{description.replace(chr(9), ' ')}\n"
    )
    with open(path, "a", encoding="utf-8") as f:
        f.write(line)


def read_results_tsv(path: str) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    rows = []
    lines = p.read_text(encoding="utf-8").splitlines()
    for line in lines[1:]:
        if not line.strip():
            continue
        commit, valid_p, test_p, wall, status, desc = line.split("\t", 5)
        rows.append({
            "commit": commit,
            "valid_primary": float(valid_p),
            "test_primary": float(test_p),
            "wall_seconds": float(wall),
            "status": status,
            "description": desc,
        })
    return rows


# ----------------------------------------------------------- concepts.json
def load_concepts(path: str) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    return json.loads(p.read_text(encoding="utf-8"))


def save_concepts(path: str, concepts: list[dict]) -> None:
    Path(path).write_text(json.dumps(concepts, indent=2), encoding="utf-8")


# --------------------------------------------------------------- runs.jsonl
def append_jsonl(path: str, record: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def read_jsonl(path: str) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


# ------------------------------------------------------------- submission -
_MAKER_SCRIPT = r"""
# Builds the submission from the AGENT'S OWN model, by running the exact code
# path the harness scores: baseline.run_fm().
#
# It does NOT re-implement training, and it does not name any model class. The
# kit's `submit.py --make` does both -- it hardcodes `B.FM(dim, k=16, lr=0.001)`
# and its own 40-epoch loop -- which fails two different ways once an agent is
# editing baseline.py:
#
#   1. LOUD: the agent renames the class (FM -> FFM) and --make dies with
#      AttributeError. Observed on the first real run.
#   2. SILENT, and far worse: the agent keeps a class called FM but changes k,
#      lr, epochs, or anything else inside run_fm. --make then succeeds, writes
#      a perfectly valid CSV, and submits the ORIGINAL baseline configuration.
#      Nothing anywhere reports a problem; the score is just wrong.
#
# run_fm returns metrics, not per-row scores, so we capture the scores on their
# way into evaluate(). run_fm's last act is
# `evaluate(ute, yte, m.predict(Xte))` -- those are exactly the test-split
# predictions, in row order. Wrapping evaluate is the only way to get them
# without requiring the agent to change run_fm's return signature, which would
# be one more contract for it to break.
import sys
import baseline as B
from data import load
from submit import write_submission

split, out_path, data_dir, seed = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
splits = load(data_dir)
rows = splits[split]

captured = []
_real_evaluate = B.evaluate


def _recording_evaluate(users, labels, scores, *a, **kw):
    captured.append((len(scores), list(scores)))
    return _real_evaluate(users, labels, scores, *a, **kw)


B.evaluate = _recording_evaluate
try:
    B.run_fm(splits, seed=seed, verbose=False)
finally:
    B.evaluate = _real_evaluate

# Match by row count: valid is 124,909 rows and test is 170,588, so the split
# is unambiguous. Last match wins -- run_fm evaluates validation every epoch,
# and the final call for a split is the one made with the restored best state.
matches = [sc for n, sc in captured if n == len(rows)]
if not matches:
    sizes = sorted({n for n, _ in captured})
    raise SystemExit(
        f"run_fm never evaluated {len(rows)} rows (the {split} split). It "
        f"evaluated {sizes}. The agent's run_fm no longer scores that split, so "
        "no submission can be built from it.")
write_submission(out_path, rows, matches[-1])
print(f"wrote {out_path}: {len(rows):,d} rows (split={split}, agent's own run_fm)")
"""


def make_submission(best_exp_dir: str, repo_root: str, data_dir: str, out_path: str,
                    seed: int = 0) -> tuple[bool, str]:
    """Write and validate the test-split submission from the best experiment.

    Runs from the winning experiment's folder so `import baseline` / `import
    data` resolve to that folder's files (Python puts the executed script's
    directory first on sys.path), and drives the agent's own `run_fm` rather
    than a re-implementation of it -- see _MAKER_SCRIPT for why that matters.

    Always ends with `submit.py --check`, which is the gate that actually
    matters: a malformed or misaligned CSV scores zero regardless of the model
    behind it, and (user_id, video_id) is not a key on the test split.

    Fails loud rather than silently producing a wrong submission.
    """
    if Path(best_exp_dir).resolve() != Path(repo_root).resolve():
        shutil.copy(Path(repo_root, "submit.py"), Path(best_exp_dir, "submit.py"))
    out_abs = str(Path(out_path).resolve())
    maker = Path(best_exp_dir, "_make_submission.py")
    maker.write_text(_MAKER_SCRIPT, encoding="utf-8")

    make = subprocess.run(
        [PYTHON, maker.name, "test", out_abs, data_dir, str(seed)],
        cwd=best_exp_dir, capture_output=True, text=True, env=_child_env(),
        timeout=RUN_TIMEOUT_SECONDS,
    )
    if make.returncode != 0:
        return False, (
            "could not build a submission from the agent's run_fm:\n"
            f"{(make.stderr or make.stdout)[-2000:]}"
        )
    check = subprocess.run(
        [PYTHON, "submit.py", "--check", "--split", "test",
         "--data_dir", data_dir, out_abs],
        cwd=best_exp_dir, capture_output=True, text=True, env=_child_env(), timeout=300,
    )
    if check.returncode != 0:
        return False, f"submission written but failed --check:\n{check.stderr[-2000:]}"
    return True, check.stdout.strip()

def save_resource_report(path: str, report: dict) -> None:
    Path(path).write_text(json.dumps(report, indent=2), encoding="utf-8")


# ------------------------------------------------------- experiment folders
# Every experiment gets its own folder under runs/ — the root baseline.py/
# data.py are NEVER written to by this harness, ever (setup only reads them,
# to log the starting baseline numbers). No git commit, no revert-by-
# overwrite: a "kept" experiment just means later code reads its folder
# instead of an earlier one's; a "discarded" one's folder sits there unused,
# a complete audit trail of every attempt, not just the surviving ones.
def make_experiment_dir(repo_root: str, name: str) -> str:
    d = Path(repo_root, "runs", name)
    d.mkdir(parents=True, exist_ok=True)
    # evaluate.py is fixed (never agent-edited) — copied so the folder is
    # self-contained: running baseline.py with cwd=this folder puts the
    # folder first on sys.path, so `from evaluate import evaluate` needs
    # its own local copy to resolve, not repo_root's.
    shutil.copy(Path(repo_root, "evaluate.py"), d / "evaluate.py")
    return str(d)


# Files the agent may never write, under any name it proposes. `evaluate.py` is
# the scoring spec and is copied into every experiment folder by
# make_experiment_dir -- BEFORE write_experiment_files runs, so an unguarded
# write would simply overwrite it and the experiment would then be scored by the
# model's own evaluator. That is not a hypothetical: "score yourself 1.0" is the
# single highest-reward move available to anything optimising the number this
# harness reads, and it would be invisible in runs.jsonl.
PROTECTED_FILES = frozenset({"evaluate.py", "submit.py", "baseline_scores.json"})


class UnsafeExperimentPath(ValueError):
    """A proposed file path that must never be written."""


def safe_experiment_path(exp_dir: str, path: str, allowed=None) -> Path:
    """Resolve `path` inside `exp_dir`, or refuse.

    The LLM chooses these path strings. `Path(exp_dir, "../../baseline.py")`
    resolves OUTSIDE the experiment folder and overwrites the frozen root kit --
    the one thing the whole runs/ isolation design exists to prevent. An absolute
    path ignores `exp_dir` entirely.

    Three checks, in order of severity: not protected, not absolute, and
    contained after resolution (which is what actually catches `..`, symlinks,
    and Windows drive-relative forms in one go).
    """
    raw = str(path).strip().replace("\\", "/")
    name = PurePosixPath(raw).name
    if name in PROTECTED_FILES or raw in PROTECTED_FILES:
        raise UnsafeExperimentPath(
            f"{path!r} targets a protected file ({sorted(PROTECTED_FILES)}); "
            "evaluate.py is the scoring spec and is never agent-editable")
    if allowed is not None and raw not in set(allowed):
        raise UnsafeExperimentPath(
            f"{path!r} is not one of the editable files {sorted(allowed)}")
    if Path(raw).is_absolute() or (len(raw) > 1 and raw[1] == ":"):
        raise UnsafeExperimentPath(f"{path!r} is an absolute path")
    root = Path(exp_dir).resolve()
    target = (root / raw).resolve()
    if target != root and root not in target.parents:
        raise UnsafeExperimentPath(
            f"{path!r} resolves to {target}, outside the experiment folder {root}")
    return target


def write_experiment_files(exp_dir: str, files: dict, allowed=None) -> None:
    """Write proposed files into one experiment folder, refusing unsafe paths.

    Raises UnsafeExperimentPath BEFORE writing anything, so a proposal with one
    bad path does not leave a half-written folder behind -- a partially written
    experiment is worse than none, because it still runs and still gets scored.
    """
    resolved = [(safe_experiment_path(exp_dir, p, allowed), c) for p, c in files.items()]
    for target, content in resolved:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8", newline="\n")


# A diff big enough to bury runs.jsonl is not more informative than a truncated
# one. 400 lines is generous for a two-file change and keeps a 50-iteration log
# readable and small enough to commit.
MAX_DIFF_LINES = 400


def unified_diff(source_dir: str, exp_dir: str, editable_files: list[str]) -> str:
    """Unified diff of what this experiment changed, for the run log.

    Never raises: a missing source file (a cleaned runs/ folder) yields an empty
    side rather than killing an experiment that has otherwise succeeded. The
    diff is documentation; it must not be able to break the thing it documents.
    """
    out: list[str] = []
    for name in editable_files:
        def _read(d):
            try:
                return Path(d, name).read_text(encoding="utf-8").splitlines(keepends=True)
            except OSError:
                return []
        before, after = _read(source_dir), _read(exp_dir)
        if before == after:
            continue
        out.extend(difflib.unified_diff(before, after,
                                        fromfile=f"a/{name}", tofile=f"b/{name}", n=3))
    if not out:
        return "(no change to the editable files — this experiment ran identical code)"
    if len(out) > MAX_DIFF_LINES:
        head = out[:MAX_DIFF_LINES]
        head.append(f"\n[... {len(out) - MAX_DIFF_LINES} more diff lines elided ...]\n")
        out = head
    return "".join(out)


def read_experiment_files(exp_dir: str, editable_files: list[str]) -> dict:
    return {p: Path(exp_dir, p).read_text(encoding="utf-8") for p in editable_files}


# ------------------------------------------------------------------- index
# SQLite (stdlib, zero new dependency — a "tiny db") — a queryable index
# over the runs/ folders (iteration, concept, metrics, outcome -> exp_dir),
# not the source of truth for file content (the folders are). No per-
# experiment git commits; the only git this harness touches at all is
# `setup`'s branch creation.
_CREATE_CHECKPOINTS_SQL = """
    CREATE TABLE IF NOT EXISTS checkpoints (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        iteration INTEGER NOT NULL,
        concept_id TEXT NOT NULL,
        exp_dir TEXT NOT NULL,
        valid_primary REAL NOT NULL,
        test_primary REAL NOT NULL,
        outcome TEXT NOT NULL,
        created_at REAL NOT NULL
    )
"""


def init_store(db_path: str) -> None:
    con = sqlite3.connect(db_path)
    try:
        con.execute(_CREATE_CHECKPOINTS_SQL)
        con.commit()
    finally:
        con.close()


def save_checkpoint(
    db_path: str, iteration: int, concept_id: str, exp_dir: str,
    valid_primary: float, test_primary: float, outcome: str,
) -> int:
    # Self-healing: ensures the table exists on every call (cheap,
    # IF-NOT-EXISTS) rather than trusting every entry point to have called
    # init_store() first. Bit us for real: cli.py setup does call it, but
    # graph.py's bootstrap node (the path a bare Studio Submit takes) didn't
    # — every Studio-only run crashed here with "no such table: checkpoints"
    # after a real, billed LLM call had already succeeded. Not repeating
    # that class of bug for the next thing that reads store_path.
    con = sqlite3.connect(db_path)
    try:
        con.execute(_CREATE_CHECKPOINTS_SQL)
        cur = con.execute(
            "INSERT INTO checkpoints (iteration, concept_id, exp_dir, valid_primary, "
            "test_primary, outcome, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (iteration, concept_id, exp_dir, valid_primary, test_primary,
             outcome, time.time()),
        )
        con.commit()
        return cur.lastrowid
    finally:
        con.close()


def load_checkpoint_dir(db_path: str, checkpoint_id: int) -> str:
    con = sqlite3.connect(db_path)
    try:
        con.execute(_CREATE_CHECKPOINTS_SQL)
        row = con.execute(
            "SELECT exp_dir FROM checkpoints WHERE id = ?", (checkpoint_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"no checkpoint {checkpoint_id} in {db_path}")
        return row[0]
    finally:
        con.close()
