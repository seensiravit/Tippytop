# Architecture

What every folder is, why it exists, and which rules must not be broken.

This repo is a **monorepo with two lanes** that share one frozen scoring script.

```
                        evaluate.py   ← frozen, ONE copy, the task spec
                       ▲            ▲
                       │            │
          autoresearch_lg/        src/tippytop/
          the agent               the model library
          writes its own code     humans test hypotheses
          50 iters, autonomous    registry + tests, minutes
                       │            │
                       └── findings ┘
                        (as prompt text)
```

The agent is the deliverable. The library is where a person proves an idea works
quickly; what it measures then goes into the agent's prompts as *context* — never
as a fixed schedule, which would score as a human-authored curriculum rather than
autonomous research.

---

## The frozen kit — repo root

The organizers' starter kit, kept at the root as **one copy**.

| File | Role |
|---|---|
| `evaluate.py` | **FROZEN. Never edit.** GAUC / nDCG@5 / primary. The entire task spec. |
| `data.py` | Loader, the fixed date splits, feature encoding. The agent edits *copies*. |
| `baseline.py` | FM / popularity / random. The FM row is what we must beat. Agent edits *copies*. |
| `submit.py` | Builds and validates the submission CSV. |
| `baseline_scores.json` | Published scores, seed variance, convergence constants. |
| `ablation_features.py` | The organizers' own feature ablation. |

Both lanes read these same files. `autoresearch_lg` seeds every experiment folder
from them; `tippytop.kit` puts the root on `sys.path` and re-exports `load`,
`encode`, `evaluate`.

> **Why one copy matters.** The two merged branches each shipped their own kit. Two
> `evaluate.py` files in a repo whose task is *defined* by `evaluate.py` is a
> correctness hazard — they drift, and nobody can say which one scored a result.
> The copies were verified logic-identical (only Chinese vs English comments and
> help strings differed); the translated one was kept.

---

## Lane 1 — `autoresearch_lg/` (the agent)

A LangGraph agent that runs the ML iteration loop on its own: propose a change,
run it, score it, decide what to try next, repeat until convergence.

```
bootstrap → eda → propose → experiment → critic → router
                    ↑                                │
                    │        error + retries left ───┘
                    │                                │
                    └─── continue ─ check_convergence ─→ finalize → END
```

| File | Responsibility |
|---|---|
| `graph.py` | The main loop and the **router** — reads the critic's verdict, sets the next mode (tune / expand / pivot), applies escalation caps. |
| `propose.py` | The "think" stage: builds the prompt, calls the LLM, validates the returned source parses. |
| `context.py` | `build_context` (run history) and `retrieve_options` (`HEADROOM` — candidate directions, offered as options, never as a schedule). |
| `experiment.py` | The "do" stage: apply, train, score. Every step has a failure branch. |
| `critic.py` | The "judge" stage: compare to best, keep or revert, classify, write the run log. |
| `bootstrap.py` | `CONFIG_DEFAULTS` (model, iteration caps, ε) and `CONSTRAINTS` (the system prompt). |
| `state.py` | `ResearchState` — the one object threaded through every node. |
| `tools.py` | Subprocess, git, results.tsv, checkpoints. Non-LLM building blocks. |
| `cli.py`, `dashboard.py` | Entry points and the run dashboard. |

**Router modes.** `improved` → *tune* (refine the same concept). `improved` at the
tune cap → *expand* (adjacent concept). `failed` → *pivot* (abandon it).
`error` → retry, then pivot. The router sets the *mode*; the LLM decides *what* to
try. Keeping that split is what makes the run autonomous rather than scripted.

**Isolation.** No git in the loop, and root `baseline.py` / `data.py` are never
written to. Each experiment gets its own folder under `runs/`, so "revert" just
means not reading that folder again.

---

## Lane 2 — `src/tippytop/` (the model library)

Where a human tests a hypothesis in minutes. Every model implements the same
two-method contract, so the runner, submission tooling and leaderboard work
unchanged.

```python
model.fit(dataset)                      # train
scores = model.predict(dataset, split)  # one float per row, in row order
```

| Path | Contents |
|---|---|
| `kit.py` | Bridge to the root kit. The only place the kit is imported from. |
| `config.py` | Paths, baseline constants, convergence parameters. |
| `data/` | `dataset.py` (load + encode once), `features.py`, `sequences.py`. |
| `losses/ranking.py` | `listwise_softmax_grad`, `bpr_grad`, `hybrid_grad`. Model-agnostic — they return the gradient wrt logits. |
| `models/` | One file per model, each `@register("name")`. `fm`, `popularity`, `random_model`, `fm_rank`, `ensemble`. |
| `training/runner.py` | The shared load → fit → score → log loop. |
| `submission.py` | Write and validate submission CSVs. |
| `cli.py` | `python -m tippytop run|submit|check|score` |
| `agent/` | **An earlier agent** (linear loop, Gemini). Kept for offline `--llm mock` testing during development — **remove before submission**, see *Decisions*. |

Adding a model is one file plus one import line in `models/__init__.py`.

---

## Supporting directories

| Path | Contents |
|---|---|
| `tests/` | Both suites. `pytest tests/ -q` — 34 tests. |
| `results/` | `leaderboard.md` (every measured run) and `submissions/`. |
| `docs/` | `tutorial.md`, `project-structure.md`, and `kit/` — the organizers' own kit docs. |
| `scripts/` | `download_data.{sh,ps1}`, experiment wrappers. |
| `experiments/configs/` | One YAML per run, for reproducibility. |
| `runs/` | Agent experiment folders. Gitignored. |
| `KuaiRand-Pure/` | The dataset, ~194 MB. Gitignored — run the download script. |

---

## Rules

1. **Never edit `evaluate.py`.** It is the scoring spec. Import it via
   `tippytop.kit` or let the agent's harness call it.
2. **Never tune against `test`.** Iterate on `valid`; `test` is for reporting the
   final pick. The hidden test set is scored once.
3. **Δvalid > 0.002 or it did not happen.** FM's seed std is 0.0008; over many
   iterations, best-of-N on that noise manufactures apparent gains.
4. **Compare like with like.** A ranking loss requires grouped batching, which
   costs ≈ −0.0023 on its own. Measure it against a pointwise control using the
   *same* batching, never against the raw FM baseline — otherwise a correct
   implementation reads as a failure. (Measured; see `results/leaderboard.md`.)
5. **Log every run** in `results/leaderboard.md`.
6. **The agent chooses what to try.** Give it context and findings, not a schedule.

---

## Environment

Managed with `uv`. One `pyproject.toml` declares both packages.

```bash
uv sync                       # or: uv pip install -e ".[dev]"
uv pip install -e ".[models]" # optional: LightGBM + scikit-learn
bash scripts/download_data.sh # or: powershell scripts/download_data.ps1
```

Keys go in `.env` (gitignored) — see `.env.example`.

`[models]` is optional on purpose: the numpy-only path still works with no extra
install, but the organizers permit any open-source library, and LightGBM's
`lambdarank` maps query groups directly onto users.

---

## Decisions

### Two agents during development — one at submission

`autoresearch_lg/` is **the** agent and the deliverable. `src/tippytop/agent/` is
an earlier implementation (linear loop, Gemini) that is **deliberately retained
during development**, for two reasons:

- **Offline testing.** It runs the full loop with `--llm mock` — no API key, no
  tokens. The LangGraph harness has no equivalent: only `cli graph` and
  `cli dashboard` run without a key, and neither exercises the loop. That makes
  it the cheap way to smoke-test plumbing changes.
- **A working fallback**, if the LangGraph harness breaks close to the deadline.

> ### ⚠️ Remove it before submitting
>
> This is a decision, not an open question. The deliverable is *"an autonomous ML
> research agent"* — a repo containing two of them, with two loops, two LLM
> providers and two prompt sets, makes a judge guess which one produced the
> result. It also drags `GEMINI_API_KEY` and a second dependency path into the
> setup instructions for no benefit.
>
> **Do this once the LangGraph agent has produced the final submission run:**
>
> ```bash
> git rm -r src/tippytop/agent
> git rm tests/test_agent_convergence.py tests/test_agent_end_to_end.py \
>        tests/test_agent_gemini_parse.py tests/test_agent_guard.py \
>        tests/test_agent_journal.py tests/test_agent_parsing.py \
>        tests/test_agent_sandbox.py
> git rm docs/agent.md
> ```
>
> Then two edits: delete `src/tippytop/cli.py` lines 130–131 (the
> `register_agent_subparser` import and call at the end of `build_parser`), and
> drop the `GEMINI_API_KEY` block from `.env.example`. Finally:
>
> ```bash
> uv run pytest tests/ -q     # must still pass (34 → 27 tests)
> ```
>
> `src/tippytop/models/` and `losses/` stay — those are the manual-experiment lane
> and are referenced by the write-up. Only `agent/` goes.
>
> Removing it is a small commit. Un-removing it at 2am is not — which is exactly
> why it stays until the final run is in hand, and not a day longer.

### Libraries are open — matching the brief, not restricting past it

The problem statement names LightGBM twice as in scope (§2.3 *In scope*; §2.4
*Resource policy*: "use any open-source library … The agent is expected to draw on
whatever published methods it can find"). The only hard rules are **no external
training data** and **no hidden-test access**.

So `CONSTRAINTS` in `bootstrap.py` offers numpy, scipy, scikit-learn and LightGBM,
and `pyproject.toml` carries them behind the optional `[models]` extra. **Torch is
excluded on time budget, not policy** — CPU training over 1.1M rows does not fit
the 10-minute per-experiment cap.

If you change one, change both: the extra controls what is *installed*, the prompt
controls what the agent will *reach for*.
