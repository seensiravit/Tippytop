# Tippytop

Tippytop is an autonomous ML research agent for within-user short-video ranking on
KuaiRand-Pure. The language model does not choose from a hidden list of canned experiments. On each
iteration it forms a hypothesis, writes a complete Python experiment, runs that source in an isolated
CPU sandbox, observes public-validation metrics, reflects, and writes the next experiment.

The benchmark objective is the mean of GAUC and nDCG@5 for `long_view`. Relative ordering matters
only among impressions belonging to the same user.

## Why This Is An Agent

Each research iteration is authored by the configured LLM and must implement:

```python
def fit(train_rows, seed):
    ...
    return model  # or (model, json_safe_metadata)


def predict(model, rows):
    ...
    return scores
```

`fit` receives a pandas DataFrame containing training labels and training-only auxiliary outcomes.
`predict` receives a feature-only DataFrame containing date, IDs, tab, duration, hour, and timestamp.
The trusted host, not generated code, computes validation metrics with the frozen evaluator. Exact
source, SHA-256 hash, unified diff, model checkpoint, metrics, LLM responses, errors, repairs, and
reflection are retained for every iteration.

There is no experiment catalogue and no deterministic experiment fallback. If generated source
fails, its traceback is returned to the LLM for up to two bounded code-repair attempts. If generation is disabled,
Tippytop reproduces the baseline and stops honestly.

## Architecture

```text
CLI
 |
 +-- doctor: environment, data, evaluator, disk, LLM endpoint
 |
 +-- trusted host agent
       |
       +-- random harness check and official FM reproduction
       +-- train/public-validation summary for the LLM
       +-- LLM hypothesis + complete Python source
       +-- AST contract and import validation
       +-- bubblewrap worker (network off, read-only runtime, bounded CPU/RAM)
       |     +-- smoke fit/predict on a bounded sample
       |     +-- full fit(train only)
       |     +-- predict(feature-only validation rows)
       +-- isolated checkpoint replay and frozen host evaluation
       +-- reflection, source diff, convergence, transaction recovery
       +-- one-time final test scoring and submission validation
```

Important modules:

| Path | Responsibility |
|---|---|
| `src/tippytop/agent.py` | Run lifecycle: preflight, harness, baseline, search, and finalization |
| `src/tippytop/search/` | Search lifecycle, iteration execution/repair, records, and crash recovery |
| `src/tippytop/llm/` | OpenAI transport, stage orchestration, prompts, and protocol types |
| `src/tippytop/generated.py` | Strict experiment schema, source hashing, and AST checks |
| `src/tippytop/runtime/` | Trusted runner, bubblewrap sandbox, and generated-code worker |
| `src/tippytop/research/` | Public helpers, rich data boundary, context, plans, and LLM contract |
| `src/tippytop/models/` | Reusable linear, MF, FM, FFM, sampling, and tree components |
| `src/tippytop/experiments/` | Trusted reference trainers and checkpoint scoring |
| `src/tippytop/submission/` | Transactional finalization, CSV checking, and report rendering |

The reusable model implementations are a research library, not a candidate list. Generated source
may import them or independently use NumPy, SciPy, scikit-learn, or LightGBM.

## Isolation And Leakage Prevention

- Generated `fit` sees only training rows. Training-only watch-time and engagement outcomes may be
  used as auxiliary supervision or to fit causal, train-derived aggregates.
- Generated `predict` sees only `date`, IDs, `tab`, `duration_ms`, `hourmin`, and `time_ms`.
- The sandbox artifact contains labeled training data and feature-only public validation data. Raw
  CSV files and validation labels are not mounted.
- Generated workers never compute benchmark metrics. The trusted host loads validation truth only
  after the sandbox exits and replay-verifies the serialized checkpoint first.
- Test aggregates, labels, and metrics never enter generation or reflection prompts.
- Random test sanity is a separate harness diagnostic and cannot affect model selection.
- The final model is selected only by public-validation primary score.
- Test scoring is idempotent and occurs once after selection.
- Bubblewrap disables network access and exposes a read-only Python/runtime tree plus one writable
  experiment directory.
- AST checks reject filesystem, subprocess, dynamic execution, introspection, and unsafe imports
  before sandbox execution. Bubblewrap remains the primary security boundary.
- API keys are never persisted. Resume reacquires a key from `TIPPYTOP_LLM_API_KEY`.

## Setup

Requirements: Linux, `uv`, and `bubblewrap` (`bwrap`). No GPU is required.

```bash
uv venv --python 3.11
uv sync
```

Download and extract KuaiRand-Pure from the organizer's Zenodo record:

```bash
wget https://zenodo.org/records/10439422/files/KuaiRand-Pure.tar.gz
tar xzf KuaiRand-Pure.tar.gz
```

The default expected data path is `KuaiRand-Pure/data/`. Raw data, environments, run artifacts,
checkpoints, and submissions are ignored by Git.

## Technology And Data

- Development tools: OpenCode, Python 3.11, `uv`, and Git.
- API: OpenAI-compatible `/models` and `/chat/completions` endpoints served by the local
  `gemma4:e4b` model.
- Libraries: NumPy, pandas, SciPy, scikit-learn, LightGBM, pytest, and the Python standard library.
- System isolation: Linux bubblewrap namespaces plus POSIX resource limits.
- Dataset and assets: KuaiRand-Pure and the organizer-provided starter loader, frozen evaluator, and
  submission checker. No external training data or pretrained weights are used.

## LLM Configuration

Defaults used for this submission:

```text
Base URL: http://100.100.10.10:12345/v1
Model:    gemma4:e4b
```

Override them with CLI flags or environment variables:

```bash
export TIPPYTOP_LLM_BASE_URL=http://100.100.10.10:12345/v1
export TIPPYTOP_LLM_MODEL=gemma4:e4b
export TIPPYTOP_LLM_API_KEY=
export TIPPYTOP_DATA_DIR=KuaiRand-Pure/data
```

## Usage

Verify the complete environment first:

```bash
uv run tippytop doctor
```

Run autonomous research with the challenge limits:

```bash
uv run tippytop run \
  --max-iterations 50 \
  --max-hours 6 \
  --epsilon 0.002 \
  --patience 3 \
  --experiment-timeout 1800 \
  --no-finalize
```

Use `--no-finalize` while developing so test is not scored. Resume an interrupted run:

```bash
uv run tippytop resume runs/<run-id> --no-finalize
```

Start a new run with only sanitized validation lessons and source failures from an earlier
validation-only run:

```bash
uv run tippytop run --learn-from runs/<prior-run-id> --no-finalize
```

This does not inherit the prior checkpoint, iteration count, convergence state, or any test data.

After validation selection is final, score test once and validate the submission:

```bash
uv run tippytop submit --run runs/<run-id>
```

Baseline-only verification without LLM calls:

```bash
uv run tippytop run --offline --no-finalize
```

## Run Artifacts

```text
runs/<run-id>/
  config.json
  state.json
  doctor.json
  random_sanity.json
  dataset_summary.json
  research_schema.json
  research_data.pkl
  prior_research.json       # only when --learn-from is used
  events.jsonl
  iterations.jsonl
  transactions/<iteration>.json
  experiments/<iteration>.json
  diffs/<iteration>.diff
  checkpoints/<experiment>/
    experiment.py
    model.pkl
    manifest.json
    valid_scores.npy
  final_submission.csv
  results.json
  report.md
```

State, source, submission, report, and JSON artifacts are written atomically. Each completed
iteration first commits a transaction; startup reconstructs state and the iteration log from those
transactions after interruption.

## Measured Results

The data harness and trusted FM reproduction were measured on the full KuaiRand-Pure split:

| Result | GAUC | nDCG@5 | Primary |
|---|---:|---:|---:|
| Random validation | 0.4993 | 0.4675 | 0.4834 |
| Random test | 0.4996 | 0.4511 | 0.4753 |
| Reproduced FM validation | 0.6671 | 0.5358 | 0.6015 |
| Best LLM-authored experiment validation | 0.6423 | 0.5238 | 0.5831 |
| Selected validation checkpoint | 0.6671 | 0.5358 | 0.6015 |
| One-time test for the selected checkpoint | 0.6621 | 0.5286 | 0.5953 |

The autonomous loop correctly retained the trusted FM checkpoint because none of its generated
experiments improved validation. Its one-time test primary is `0.5953407`, which is `+0.0007407`
above the published FM test primary of `0.5946`. The validation delta is `+0.0000`, so this run does
**not** clear the challenge's `+0.002` target. This is reported as a limitation rather than
overstated.

The finalized source-writing run is `runs/20260829T083524Z-seed0`. It converged after three measured
iterations in 1,031.9 seconds, used 41,925 LLM tokens, used zero GPU-hours, and required zero manual
interventions. Iteration 3 failed smoke on an obsolete LightGBM argument, was repaired by the LLM,
then completed full training and checkpoint replay. The final submission contains 170,588 rows and
passed the organizer checker.

## Verification

```bash
uv run pytest
uv run python -m compileall -q src tests
git diff --check
uv run tippytop doctor
```

The suite covers source parsing and safety, JSON-mode repair and audit retention, same-user pair
sampling, parametric and tree checkpoints, convergence semantics, transaction recovery, finalization
idempotence, rich feature-only validation isolation, namespace sandbox execution, generated
checkpoint replay, source integrity, and finite score validation.

## Limitations

- The generated experiments did not improve the trusted FM validation checkpoint; validation delta
  is `+0.0000`, below the `0.002` target.
- Generated code quality is bounded by the local LLM and the experiment time budget.
- Bubblewrap makes generated execution Linux-specific.
- Generated models are pickled and should only be loaded from their trusted local run directory.
- Rich auxiliary outcomes are available only during training. Generated code must turn them into
  causal histories or train-fitted aggregates that can be reproduced from prediction-time fields.
- The AST policy is defense in depth, not the sole security boundary; OS namespace isolation remains
  mandatory.

## Devpost Summary

Tippytop turns a local OpenAI-compatible model into an auditable recommender-system researcher. It
writes real experiment code rather than selecting canned configurations, executes each idea inside a
networkless CPU sandbox, measures only public validation, repairs failures from tracebacks, and keeps
the exact source-level research trail. The project reproduces the official benchmark, improves the
published test baseline by `+0.0007407`, creates a checker-valid submission, and reports that its
LLM-authored experiments did not beat the validation baseline rather than overstating the outcome.
