# The Autonomous ML Research Agent

Our answer to TechJam Track 2: an LLM-driven agent that runs the ML iteration loop
on its own — it **writes the code** for each attempt, runs it, scores it, reflects,
and iterates, aiming to beat the FM baseline on KuaiRand-Pure. Lives in
`src/tippytop/agent/`.

## The idea (AIDE-style code optimization)

Each iteration the LLM writes a full runnable `solution.py`. A sandboxed subprocess
runs it on the **valid** split; the frozen `evaluate` scores it; the **best-by-valid**
solution is kept and improved next round. The hidden **test** set is touched exactly
once, at the very end, by the harness — never by the LLM.

```
iter 0  SEED     hardcoded FM baseline script (no LLM)     -> reproduce 0.5946
iter n  IMPROVE  LLM edits the best solution               -> run on valid, keep if better
        DEBUG    (if last run failed) LLM fixes it, given the traceback
...     stop on  ε=0.002/N=3 convergence | 50 iters | 6h
FINALIZE         run the valid-best on TEST once -> final submission + report
```

## How to run

```bash
# offline, zero network — proves the machinery (default backend is the mock)
python -m tippytop agent --llm mock --max-iters 3

# live, with Gemini — put your key in .env (copy from .env.example), then:
python -m tippytop agent --llm gemini --max-iters 50
```

The key is read from `.env` (`GEMINI_API_KEY=...`, git-ignored) or the
`GEMINI_API_KEY` env var if already set (the shell wins over `.env`).

Flags: `--llm {mock,gemini}`, `--max-iters`, `--wall-hours`, `--iter-timeout`,
`--temperature`, `--run-id`, `--final-out`. Model defaults to **`gemini-3.5-flash-lite`**
(the API retired `gemini-2.5-flash-lite` for new keys). The key is read from the
`GEMINI_API_KEY` env var and is **stripped from the subprocess env** so generated
code can never read it.

## What each run produces (the deliverables)

Under `results/runs/<run_id>/` (git-ignored):
- `iter_<n>/solution.py`, `scores.csv`, `stdout.txt`, `stderr.txt`, `diff.patch`
- `journal.jsonl` — one record per iteration: hypothesis, metrics (valid only),
  error/recovery, tokens, wall-clock
- `report.md` — human-readable summary: best valid primary, **final test primary vs
  FM/oracle**, total tokens, wall-clock, **manual interventions**, stop reason
- final submission CSV under `results/submissions/agent_<run_id>_test.csv`

## Architecture (`src/tippytop/agent/`)

| Module | Responsibility |
|---|---|
| `orchestrator.py` | the control loop + state machine + finalize |
| `config.py` | `AgentConfig` (budgets, convergence, paths) |
| `llm/base.py` · `mock.py` · `gemini.py` | client ABC; scripted offline mock; stdlib-only Gemini REST |
| `prompts.py` | system / improve / debug prompts (valid-only feedback) |
| `parsing.py` | extract hypothesis + the last ```python fence |
| `contract.py` | the solution contract + the iter-0 seed script |
| `guard.py` | static AST/regex block on test-split access & network imports |
| `sandbox.py` | subprocess exec with timeout + tree-kill (Windows-safe) |
| `scoring.py` | `read_submission` → `evaluate` (valid-only wrapper) |
| `convergence.py` | ε/N convergence + iteration/wall budget |
| `journal.py` | JSONL trace + markdown report |
| `cli.py` | the `agent` subcommand |

Reuses the existing pipeline unchanged: `tippytop.kit` (frozen load/encode/evaluate),
`submission.write_submission`/`read_submission`, `training.runner`, `config`.

## Test-set walling (why the score is trustworthy)

Four independent guarantees: (1) the loop only ever runs `--split valid`; (2) prompts
carry valid-only metrics via a single choke-point; (3) the guard blocks generated code
from reading `splits['test']`; (4) test is scored once at finalize, after the loop,
never fed back. The end-to-end test asserts all four.

## Robustness

A failed iteration (guard violation, timeout, runtime error, malformed reply, invalid
output) becomes a **recovery event**, not a crash: the next iteration is a DEBUG prompt
with the traceback, and the run continues. Verified by `tests/test_agent_end_to_end.py`
(a deliberately broken solution → recovery → keep-best).

## Status & next steps

Done: the full loop, mock-tested end-to-end (26 tests), and the live Gemini client
(verified reproducing valid 0.6019 / test 0.5957, tokens & wall-clock logged, 0 manual
interventions). **Not yet done** (next phase): prompt tuning and improvement directions
to actually push valid primary above the baseline — a ranking loss (BPR/listwise) is the
highest-value target. See `task.md`.

## Tests

```bash
python -m pytest tests/ -q          # 26 pass; agent tests need no network
```
