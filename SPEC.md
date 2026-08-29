# Tippytop Autonomous ML Research Agent

**Status:** Implemented and validated
**Hackathon:** TikTok TechJam 2026, Track 2
**Benchmark:** KuaiRand-Pure
**Last updated:** 2026-08-29

## 1. Purpose

Tippytop is a local autonomous ML research agent for recommendation ranking. On every iteration the
configured language model forms a hypothesis and writes a complete Python experiment. The system
validates and executes that source in a bounded sandbox, evaluates it on public validation with the
frozen organizer metric, reflects on the result, and continues until convergence or a resource limit.

The language model is not a selector over hardcoded experiment candidates. Trusted code supplies the
benchmark harness, safety boundary, evaluator, baseline reproduction, checkpoint protocol, and final
submission path. The LLM supplies the experiment implementation.

## 2. Goals

The system must:

- Create and synchronize its environment with `uv`.
- Verify data, evaluator, disk, bubblewrap, and the OpenAI-compatible LLM service.
- Reproduce published random and pointwise-FM references.
- Let the LLM write actual experiment source for each research iteration.
- Prevent generated code from reading validation/test labels through its function interface.
- Execute generated source without network access or writable access to raw data or project source.
- Persist hypotheses, exact source, source hashes, unified diffs, metrics, errors, repairs, and
  reflections.
- Select checkpoints only by public-validation primary score.
- Stop on convergence, iteration budget, or six-hour wall-clock budget.
- Resume from atomically persisted state.
- Score test once after selection and produce a checker-valid submission.
- Report performance and resource use without overstating statistical significance.

The challenge performance target remains a validation gain greater than `0.002` over the official FM
reference. It is a research target, not a condition for claiming that unmeasured work succeeded.

## 3. Non-Goals

- Modifying `evaluate.py` or its metric conventions.
- Using test results in prompts, reflection, model selection, or convergence.
- Giving generated code shell, network, raw-data, or main-source write access.
- Letting generated source change the host evaluator or checkpoint-selection logic.
- Requiring a GPU.
- Pretending offline deterministic experiments are autonomous LLM research.
- Running optional larger KuaiRand variants before the Pure benchmark is reliable.

## 4. Benchmark Contract

The frozen starter evaluator is authoritative.

| Property | Contract |
|---|---|
| Task | Rank each user's own logged impressions |
| Label | Native binary `long_view` |
| Metrics | GAUC and nDCG@5 |
| Primary | `(GAUC + nDCG@5) / 2` |
| Train | 2022-04-08 through 2022-04-21 |
| Validation | 2022-04-22 through 2022-04-28 |
| Test | 2022-04-29 through 2022-05-08 |
| Convergence epsilon | `0.002` |
| Convergence patience | 3 non-significant iterations |
| Maximum iterations | 50 |
| Maximum wall time | 6 hours |

Published references:

| Model | Split | GAUC | nDCG@5 | Primary |
|---|---|---:|---:|---:|
| Random | Validation | 0.4993 | 0.4675 | 0.4834 |
| Random | Test | 0.4996 | 0.4511 | 0.4753 |
| FM | Validation | 0.6674 | 0.5357 | 0.6016 |
| FM | Test | 0.6610 | 0.5282 | 0.5946 |
| Oracle | Test | 1.0000 | 0.7289 | 0.8645 |

Only within-user score order matters. Submission rows preserve the exact loader order and use
`row_id,user_id,video_id,score`; `(user_id, video_id)` is not unique.

## 5. Data And Test Isolation

The trusted host loads the official chronological splits. It creates a sandbox research artifact
containing a labeled training DataFrame and a feature-only public-validation DataFrame. The raw data
directory is not mounted because the organizer's later CSV physically contains both validation and
test labels.

Generated source has this interface:

```python
def fit(train_rows, seed):
    ...


def predict(model, rows):
    ...
```

`fit` receives a pandas DataFrame with prediction-time fields, the target, and training-only
auxiliary outcomes:

```text
date, user_id, video_id, author_id, tab, duration_ms, hourmin, time_ms,
long_view, play_time_ms, is_click, is_like, is_follow, is_comment,
is_forward, is_hate, profile_stay_time, comment_stay_time, is_profile_enter
```

`predict` receives a feature-only pandas DataFrame with:

```text
date, user_id, video_id, author_id, tab, duration_ms, hourmin, time_ms
```

The sandbox worker never owns validation labels. It validates score shape and finiteness, serializes
the model and scores, and exits. The trusted host then replays that checkpoint in a separate sandbox,
requires bit-exact validation scores, and only then loads validation truth and invokes the frozen
evaluator. Generated source never receives validation labels, the evaluator, or the raw data path.

The dataset summary sent to the LLM contains train and validation aggregates only. Test label
aggregates are excluded. Random test sanity is persisted separately as a harness diagnostic and is
never placed in generation or reflection context.

Finalization is idempotent. Once `test_evaluated` is persisted, resume and submit reuse the existing
submission rather than evaluate test again.

## 6. Environment

The project targets Python 3.11 or newer and is managed by `uv`:

```bash
uv venv --python 3.11
uv sync
```

Required runtime dependencies are NumPy, pandas, scikit-learn, and LightGBM. Bubblewrap is an
OS-level runtime requirement for generated code. The baseline remains CPU-only and no GPU framework
is used.

Raw data, `.venv`, checkpoints, generated submissions, and run artifacts are ignored by Git.

## 7. LLM Configuration

Defaults:

```text
TIPPYTOP_LLM_BASE_URL=http://100.100.10.10:12345/v1
TIPPYTOP_LLM_MODEL=gemma4:e4b
TIPPYTOP_LLM_API_KEY=
TIPPYTOP_DATA_DIR=KuaiRand-Pure/data
```

The HTTP client uses `/models` and `/chat/completions`, OpenAI-compatible JSON mode for generated
source, finite timeouts, bounded retries, and exponential backoff. Authorization is added only to
HTTP headers. Persisted configuration is redacted; resume reacquires the key from the environment.

## 8. Generated Experiment Schema

The LLM returns exactly one JSON object:

```json
{
  "hypothesis": "A falsifiable claim about within-user ranking.",
  "expected_effect": "The expected change in validation GAUC/nDCG@5.",
  "source": "Complete Python module as a JSON string."
}
```

Unknown or missing fields are rejected. The source must define exactly one top-level two-positional-
argument `fit` and `predict`; parameter names and harmless defaults are not semantically relevant.
Source is limited to 100,000 characters.

Allowed imports cover most safe standard-library modules, NumPy, pandas, SciPy, scikit-learn,
LightGBM, `tippytop.models`, and `tippytop.research`. Relative imports, other Tippytop internals,
filesystem/process/runtime-escape modules, wildcard imports, and dunder access are rejected.

AST checks also reject:

- `open`, dynamic import, `eval`, `exec`, and `compile`.
- Reflection helpers such as `getattr`, `globals`, `locals`, and `vars`.
- Filesystem, subprocess, and unsupported module imports.
- Invalid or asynchronous entry points.

The returned model must be pickleable. `fit` may return either the model or
`(model, json_safe_metadata)`. `predict` must return one finite numeric score per row.

## 9. Sandbox Contract

Bubblewrap launches each generated worker with:

- New network, PID, IPC, UTS, and cgroup namespaces, with no network access.
- A read-only system runtime, virtual environment, Tippytop package, and frozen evaluator.
- Read-only access to the explicit sanitized research artifact or selected checkpoint.
- One writable experiment/workspace directory.
- A private `/tmp`, `/proc`, and `/dev`.
- A sanitized environment with no API key.
- Fixed BLAS thread limits.
- An RLIMIT CPU timeout and 16 GiB address-space cap.
- A parent-controlled wall timeout.

Source is validated and hashed before execution. A bounded real-data smoke fit/predict must pass
before full training. The host verifies that persisted source remains byte-identical afterward.
Prediction revalidates source and its hash before loading the model. Pickle files are loaded only
inside the sandbox from a trusted local run directory.

## 10. Lifecycle

### 10.1 Preflight

`doctor` checks Python/NumPy, required files and headers, frozen evaluator hash, disk space,
bubblewrap availability, LLM connectivity, and configured model availability. Offline mode skips LLM
connectivity only.

### 10.2 Harness Validation

Five deterministic random seeds must reproduce published validation/test sanity within `0.001`.
Failure stops the run before research metrics are trusted.

### 10.3 Baseline Reproduction

The trusted pointwise FM trains using train and validation only. Validation primary must be within
`0.003` of `0.6016`; otherwise the run fails diagnostically.

### 10.4 Dataset Analysis

The host records rows, users, items, positive rate, dates, per-user impression/positive
distributions, and all-negative/all-positive/discriminative-user counts for train and validation.
Only aggregate data enters the prompt.

### 10.5 Source Generation

The LLM receives the benchmark contract, generated-code interface and helper signatures, sandbox
limits, organizer-published dead ends and headroom guidance, current validation best, exact source of
the current generated best when one exists, and bounded recent source/outcome/recovery history. It
returns a hypothesis, expected effect, and complete source. A malformed response receives up to two
bounded schema/source corrections. Every raw response and its token/model provenance remains in the
iteration record for diagnosis.

Repeated source hashes are rejected rather than re-evaluated.

### 10.6 Execution And Repair

The worker first smoke-tests generated `fit`/`predict`, then calls `fit` on full train and `predict`
on feature-only validation rows. The trusted host verifies checkpoint replay and evaluates scores.
On runtime failure, timeout, invalid score, or serialization failure, the traceback is sent to the
LLM for up to two source-repair attempts. Comment-only and repeated-source repairs are rejected.
There is no deterministic model fallback. Failed repairs restore the previous validation best.

### 10.7 Reflection

The LLM receives the hypothesis, source hash, measured validation metrics or error/recovery, and the
best metrics that existed before the experiment. Reflection is logged and informs later generation;
it cannot execute code.

### 10.8 Selection And Convergence

Any strict primary-score improvement becomes the stored validation best. Only an improvement greater
than `epsilon` resets stagnation. This retains small real bests while preventing noise-sized changes
from extending the run indefinitely.

The loop stops at convergence, 50 iterations, six hours, or explicit interruption. Offline mode
reproduces the baseline and stops with `offline_baseline_only`; it does not simulate autonomy.

### 10.9 Finalization

The selected checkpoint predicts test once, the frozen evaluator records metrics, and the host writes
the exact submission order. The organizer's `submit.py --check` must pass before the run is marked
complete.

## 11. Persistence And Resume

Each run contains:

```text
runs/<run-id>/
  config.json
  state.json
  doctor.json
  random_sanity.json
  dataset_summary.json
  research_schema.json
  research_data.pkl
  prior_research.json
  events.jsonl
  iterations.jsonl
  transactions/
  experiments/
  diffs/
  checkpoints/
  final_submission.csv
  results.json
  report.md
```

JSON state uses fsync plus atomic rename. Each completed iteration first commits an atomic transaction
containing its state and record; startup reconstructs state and `iterations.jsonl` from committed
transactions. Transient generation transport failures do not consume an experiment iteration.

Every executed iteration records:

- Iteration, experiment, and parent IDs.
- Hypothesis and expected effect.
- Initial and executed source hashes.
- Exact generated source in the experiment artifact.
- Exact unified source diff in both the JSONL record and `diffs/`.
- Seed and host source revision.
- Validation GAUC, nDCG@5, primary, and checkpoint.
- Start time and duration.
- Captured output tails.
- LLM responses and token usage.
- Runtime errors and source-repair actions.
- Whether it became best or exceeded epsilon.
- Cumulative manual-intervention count.

## 12. Code Layout

```text
src/tippytop/
  agent.py
  artifacts.py
  cli.py
  config.py
  convergence.py
  doctor.py
  generated.py
  starter.py
  llm/
    client.py
    prompts.py
    protocol.py
    transport.py
  research/
    api.py
    context.py
    contract.py
    data.py
    plan.py
  runtime/
    runner.py
    sandbox.py
    worker.py
  search/
    execution.py
    iteration.py
    journal.py
    loop.py
    records.py
  submission/
    checker.py
    finalize.py
    report.py
  experiments/
    scoring.py
    trainer.py
    tree_trainer.py
  models/
    base.py
    linear.py
    matrix_factorization.py
    factorization_machine.py
    field_aware.py
    registry.py
    sampling.py
    trees/
tests/
```

The model and experiment packages are reusable trusted research components, not a hardcoded search
space. Generated code may use them or other permitted CPU libraries.

## 13. CLI

```bash
uv run tippytop doctor
uv run tippytop run --max-iterations 50 --max-hours 6 --no-finalize
uv run tippytop run --learn-from runs/<prior-run-id> --no-finalize
uv run tippytop resume runs/<run-id> --no-finalize
uv run tippytop submit --run runs/<run-id>
```

Useful run flags include endpoint/model/key overrides, data/runs paths, epsilon, patience, seed,
experiment and LLM timeouts, `--learn-from`, `--offline`, and `--no-finalize`. `--learn-from` creates
a fresh run with bounded validation-only history; it does not inherit checkpoints or convergence.

## 14. Verification

```bash
uv run pytest
uv run python -m compileall -q src tests
git diff --check
uv run tippytop doctor
```

Automated coverage includes:

- Generated schema, JSON extraction, hashing, AST allow/deny cases, and entry-point arity.
- Malformed LLM response repair and runtime source-repair parsing.
- Namespace sandbox execution, generated checkpoint replay, and source integrity.
- Feature-only validation serialization, test-summary exclusion, and host-only metric evaluation.
- Same-user pair construction and synthetic BPR ordering.
- Linear, MF, FM, FFM, random forest, histogram boosting, and LambdaRank checkpoints.
- Train-only/leave-one-out target aggregates.
- Convergence epsilon semantics.
- Transaction recovery, atomic final artifacts, one-time finalization, and secret redaction.

## 15. Acceptance Status

Implemented and verified:

- `uv` environment and installable CLI.
- Full-data random references and FM validation reproduction.
- Real LLM-authored source path with sandbox and repair contract.
- Validation-only selection and test-prompt isolation.
- Atomic artifacts, resume state, convergence, and resource accounting.
- Checker-valid final submission.
- Final source-writing run `runs/20260829T083524Z-seed0`: three measured iterations, 1,031.9 seconds,
  41,925 LLM tokens, zero GPU-hours, and zero manual interventions.
- One-time selected-checkpoint test primary `0.5953407`, `+0.0007407` above published FM test
  `0.5946`.

Open performance target:

- The reproduced FM remained validation-best at primary `0.6014695`; the best LLM-authored
  experiment reached `0.5830568`. Validation delta is therefore `+0.0000`, not beyond the requested
  `+0.002` noise band.
- The run demonstrates autonomous source generation, bounded repair, execution, replay, convergence,
  selection, and one-time finalization, but it does not satisfy the challenge's validation-improvement
  performance target.
