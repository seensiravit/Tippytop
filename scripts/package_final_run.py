"""Collect a finished run's graded artifacts into results/final_run/ — and refuse
to do it if they are not actually gradeable.

    python scripts/package_final_run.py            # check, then copy
    python scripts/package_final_run.py --dry-run  # check only

Why a script rather than a `cp` line in a README
------------------------------------------------
Four of the five files here are deliverables, and each has a way of being
present but worthless:

* ``submission.csv`` can be well-formed and misaligned. ``(user_id, video_id)``
  is not a key on the test split — 3.06% of its rows are repeated pairs — so
  alignment is checked row by row against ``row_id``. A file that fails this
  scores zero regardless of the model behind it, which is the single largest
  avoidable risk in the whole submission.
* ``runs.jsonl`` can be a smoke test. Three iterations of a mock LLM look
  exactly like a real run to ``cp``, so the iteration count is reported and a
  suspiciously short run is called out.
* ``resource_report.json`` can carry a manual-intervention count of zero
  because nothing ever incremented it. The count is cross-checked against
  ``interventions.jsonl``, and a mismatch is an error rather than a warning.
* ``concepts.json`` is the Autonomy evidence — the record of what the agent
  opened, expanded and closed on its own, with reasons.

Everything it reports, it reports before copying, so a bad run is caught while
it can still be re-run rather than after it has been committed.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "results" / "final_run"

REQUIRED = ("runs.jsonl", "resource_report.json", "submission.csv",
            "results.tsv", "concepts.json")
OPTIONAL = ("interventions.jsonl", "recovery.jsonl", "results_dashboard.html")

# Below this, a run is more likely a smoke test than a submission.
MIN_ITERATIONS = 5


class Problem(Exception):
    pass


def _load_jsonl(path: Path) -> list[dict]:
    out = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if ln:
            out.append(json.loads(ln))
    return out


def check(root: Path, strict: bool) -> dict:
    problems: list[str] = []
    notes: list[str] = []

    missing = [f for f in REQUIRED if not (root / f).exists()]
    if missing:
        raise Problem(
            "these graded artifacts do not exist yet: " + ", ".join(missing)
            + "\nThey are produced by the agent's `finalize` node, which only "
              "fires on convergence. Run the agent to completion first:\n"
              "  python -m autoresearch_lg.cli setup --tag final\n"
              "  python -m autoresearch_lg.cli run   --tag final"
        )

    runs = _load_jsonl(root / "runs.jsonl")
    report = json.loads((root / "resource_report.json").read_text(encoding="utf-8"))

    # --- the run itself -----------------------------------------------------
    n_iter = len(runs)
    notes.append(f"runs.jsonl: {n_iter} iterations")
    if n_iter < MIN_ITERATIONS:
        problems.append(
            f"only {n_iter} iterations — that reads as a smoke test, not a "
            f"submission run (expected at least {MIN_ITERATIONS})")

    outcomes: dict[str, int] = {}
    for r in runs:
        outcomes[r.get("outcome", "?")] = outcomes.get(r.get("outcome", "?"), 0) + 1
    notes.append("outcomes: " + ", ".join(f"{k}x{v}" for k, v in sorted(outcomes.items())))
    if outcomes.get("error", 0) == 0 and n_iter >= MIN_ITERATIONS:
        notes.append("note: no failed iterations — Robustness is graded on recovery, "
                     "so if nothing ever failed, say so explicitly in the write-up "
                     "rather than leaving a judge to wonder whether it was tested")

    # --- Deliverable 3 shape ------------------------------------------------
    # "Hypothesis ... the code diff applied ... resulting metrics ... any error
    # or recovery events." A log missing any of those is not the artifact the
    # brief asks for, however many iterations it has.
    real = [r for r in runs if r.get("mode") != "baseline"]
    no_diff = [r["iteration"] for r in real if not r.get("diff")]
    if real and no_diff:
        problems.append(
            f"{len(no_diff)} iteration(s) in runs.jsonl have no 'diff' field "
            f"(e.g. #{no_diff[0]}). Deliverable 3 requires the code diff applied "
            "per iteration; a run logged before diff capture was added must be re-run.")
    no_hypo = [r["iteration"] for r in real if not r.get("hypothesis")]
    if real and no_hypo:
        problems.append(f"{len(no_hypo)} iteration(s) have no 'hypothesis' "
                        "(Deliverable 3 requires it per iteration)")
    if real and not any(r.get("metrics", {}).get("valid", {}).get("GAUC") for r in real):
        problems.append("no per-metric GAUC/nDCG@5 in runs.jsonl — Deliverable 4's "
                        "results table cannot be filled from this run")

    # --- Deliverable 4 results table ----------------------------------------
    if report.get("score_dataset") is None:
        notes.append("note: no score_dataset in resource_report — either nothing "
                     "beat the baseline, or this run predates per-metric capture")
    else:
        d = report.get("test_delta_vs_baseline", {})
        notes.append("delta vs official baseline: "
                     + ", ".join(f"{k} {v:+.4f}" for k, v in d.items())
                     + f" -> score_dataset {report['score_dataset']:+.4f}")

    # --- resources ----------------------------------------------------------
    for key in ("iterations", "elapsed_seconds", "tokens_in_total", "tokens_out_total"):
        if key not in report:
            problems.append(f"resource_report.json is missing '{key}' (Deliverable 4)")
    if "elapsed_seconds" in report:
        notes.append(f"wall clock: {report['elapsed_seconds'] / 3600:.2f} h")
    if "tokens_in_total" in report:
        notes.append(f"tokens: {report['tokens_in_total']:,} in / "
                     f"{report.get('tokens_out_total', 0):,} out")

    # --- interventions ------------------------------------------------------
    ipath = root / "interventions.jsonl"
    on_disk = len(_load_jsonl(ipath)) if ipath.exists() else 0
    reported = report.get("manual_interventions")
    if reported is None:
        problems.append(
            "resource_report.json has no 'manual_interventions' field. "
            "Deliverable 3 requires the count and Impact & Relevance (20%) is "
            "scored on it — this report predates the wiring, so re-run finalize.")
    elif reported != on_disk:
        problems.append(
            f"manual_interventions says {reported} but interventions.jsonl holds "
            f"{on_disk}. One of them is wrong; do not submit either number.")
    else:
        notes.append(f"manual interventions: {reported} "
                     f"({report.get('intervention_summary', '')})")

    # --- recovery evidence --------------------------------------------------
    # Robustness is 20% and is graded on recovery, so an empty recovery log on a
    # long run is worth a second look: either genuinely nothing went wrong, or
    # the policy is not wired to the loop that is running.
    rec = _load_jsonl(root / "recovery.jsonl") if (root / "recovery.jsonl").exists() else []
    if "recovery_events" in report and report["recovery_events"] != len(rec):
        problems.append(f"resource_report says {report['recovery_events']} recovery "
                        f"events but recovery.jsonl holds {len(rec)}")
    by_action: dict[str, int] = {}
    for e in rec:
        by_action[e.get("action", "?")] = by_action.get(e.get("action", "?"), 0) + 1
    notes.append("recovery: " + (", ".join(f"{k}x{v}" for k, v in sorted(by_action.items()))
                                 if by_action else "no events recorded"))
    if "stop_reason" in report:
        notes.append(f"stopped because: {report['stop_reason']}")

    # --- the submission -----------------------------------------------------
    sub = root / "submission.csv"
    proc = subprocess.run(
        [sys.executable, "submit.py", "--check", str(sub), "--split", "test"],
        cwd=root, capture_output=True, text=True,
        env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8"})
    if proc.returncode != 0:
        problems.append("submit.py --check REJECTED the submission:\n"
                        + (proc.stdout + proc.stderr).strip()[-1500:])
    else:
        notes.append(f"submission.csv: passed --check --split test "
                     f"({sum(1 for _ in sub.open()) - 1:,} rows)")

    if problems and strict:
        raise Problem("\n\n".join(problems))
    return {"notes": notes, "problems": problems, "report": report, "runs": runs}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--dry-run", action="store_true", help="check only, copy nothing")
    ap.add_argument("--force", action="store_true",
                    help="copy even if a check failed (you will have to justify it)")
    a = ap.parse_args()
    root = Path(a.root).resolve()

    try:
        res = check(root, strict=not a.force)
    except Problem as e:
        print("REFUSED\n" + str(e), file=sys.stderr)
        return 1

    for n in res["notes"]:
        print("  " + n)
    for p in res["problems"]:
        print("  WARNING: " + p.splitlines()[0])

    if a.dry_run:
        print("\ndry run — nothing copied.")
        return 0

    DEST.mkdir(parents=True, exist_ok=True)
    copied = []
    for f in REQUIRED + OPTIONAL:
        src = root / f
        if src.exists():
            shutil.copy2(src, DEST / f)
            copied.append(f)
    print(f"\ncopied into {DEST.relative_to(ROOT)}: {', '.join(copied)}")
    print("\nnext:\n  git add results/final_run && git commit -m 'Add final run artifacts'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
