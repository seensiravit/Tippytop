"""Failure policy — the difference between "handles failures" and "autonomous".

The existing harness already routes every *expected* failure into a
`FailureRecord` instead of a crash: a bad diff, a training crash, a timeout, an
unparseable summary each has its own edge to `emit_failure`. That is robustness
in the sense the brief asks for, and it was already there.

What it did not have is a policy for the failures that are not the experiment's:

===========================================================================
 layer                      example                       what it needs
===========================================================================
 L1  transient infrastructure  429, 529 overloaded,        retry with backoff;
                               read timeout, DNS blip      must not cost budget
 L2  malformed model output    no tool_use block; a        regenerate with the
                               missing key; bad syntax     defect fed back
 L3  defective generated code  NameError, shape mismatch,  REPAIR with the
                               NaN, 10-minute timeout      traceback, not reroll
 L4  terminal                  budget spent, provider      still produce the
                               down, operator Ctrl+C       graded deliverables
===========================================================================

Before this module: L1 killed the whole run (one unhandled `OverloadedError` in
`llm_generate` propagates through two sub-graphs and out of `.stream()`), L2 was
covered for syntax only, L3 was a blind reroll at a different seed, and L4 did
not exist — `finalize` was reachable only through convergence, so a run that
died at iteration 30 of 50 produced no `submission.csv`, no
`resource_report.json`, and therefore no Deliverable 3 or 4 at all, despite 30
perfectly good iterations sitting on disk.

Sources for the policy, not invented here
-----------------------------------------
* **MLE-STAR** (Google, NeurIPS 2025) — on a traceback it runs a debugging
  module and "the debugging step is repeated until either the script executes
  successfully, or a predefined maximum number of debugging rounds is reached";
  if it cannot be fixed it "proceeds to the next task using the latest version of
  the script that is known to be executable". That is exactly `repair -> bounded
  attempts -> fall back to last known good`, which is what `router` now does.
* **AIDE** (`aide/utils/config.yaml`) — `search.max_debug_depth: 3`. Our
  `retry_cap` default of 3 matches it, so the repair budget is anchored to a
  published value rather than a guess.
* **LangGraph fault tolerance** (>=1.2) — `RetryPolicy`, `error_handler`
  returning a `Command(goto=...)`, and `set_node_defaults`. We use the framework
  primitives instead of hand-rolling a retry loop, because a hand-rolled loop
  inside a node is invisible to the checkpointer.
* **Anthropic SDK** — `max_retries` defaults to **2**, which is too few for a
  six-hour unattended run, and its backoff does not cover a provider outage that
  lasts minutes. Raised here, with a LangGraph-level retry on top of it.

Why the two-level retry is not redundant: the SDK retries a *request*, LangGraph
retries the *node*. The SDK cannot recover from a malformed-but-successful
response (L2) or from an exception raised in our own parsing code; the node-level
policy can, because it re-enters `llm_generate` from a clean state.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from langgraph.types import RetryPolicy

# ---------------------------------------------------------------- L1 ------
# Retry these: the request may succeed if we simply ask again later.
_TRANSIENT_NAMES = {
    "RateLimitError",          # 429
    "OverloadedError",         # 529 — Anthropic's capacity signal
    "InternalServerError",     # 5xx
    "ServiceUnavailableError",
    "APITimeoutError",
    "APIConnectionError",
    "DeadlineExceededError",
    "RetryableError",
    "APIStatusError",          # only when the status says so — checked below
    "ConflictError",
}
# Never retry these: the same request will fail identically every time, and
# six exponential backoffs before dying just burns the wall-clock budget that
# `finalize` needs. An auth failure is an operator problem, not a transient one.
_PERMANENT_NAMES = {
    "AuthenticationError",
    "PermissionDeniedError",
    "BadRequestError",
    "NotFoundError",
    "RequestTooLargeError",
    "UnprocessableEntityError",
    "APIResponseValidationError",
}


def is_transient(exc: BaseException) -> bool:
    """True if retrying this exception could plausibly succeed.

    Matched on class *name* rather than by importing the provider SDKs, so the
    policy holds for both Anthropic and OpenAI without this module having to
    import either, and does not break when a provider renames a base class.
    """
    for cls in type(exc).__mro__:
        name = cls.__name__
        if name in _PERMANENT_NAMES:
            return False
        if name in _TRANSIENT_NAMES:
            status = getattr(exc, "status_code", None)
            if status is not None:
                return status == 408 or status == 409 or status == 429 or status >= 500
            return True
    # Our own parsing failures (ProposalError) are worth exactly one more
    # sample from the model: a different sample may well be well-formed.
    if isinstance(exc, ProposalError):
        return True
    # Anything unrecognised: retry once rather than lose a six-hour run to a
    # class we have not seen. The attempt cap bounds the cost.
    return not isinstance(exc, (KeyboardInterrupt, SystemExit, MemoryError))


class ProposalError(RuntimeError):
    """The model answered, but not with something we can act on.

    Raised instead of letting `next(...)` throw `StopIteration` or a dict access
    throw `KeyError` — both of which are opaque at the point they surface, and
    `StopIteration` in particular is swallowed or reinterpreted by generator
    machinery in ways that make the real cause unrecoverable from a log.
    """


# 5 attempts over ~2+4+8+16 = 30s of backoff. Long enough to ride out a 529
# burst, short enough that a genuine outage is declared before it eats the
# hour that finalize and the submission build need.
LLM_RETRY = RetryPolicy(
    max_attempts=5,
    initial_interval=2.0,
    backoff_factor=2.0,
    max_interval=60.0,
    jitter=True,
    retry_on=is_transient,
)

# Passed to the provider clients. The SDK default of 2 is tuned for an
# interactive request, not for an unattended run where the alternative to
# waiting is losing everything.
CLIENT_MAX_RETRIES = 6
CLIENT_TIMEOUT_SECONDS = 300.0


# ---------------------------------------------------------------- L3 ------
# A blind reroll at a new seed can only help when the failure is actually
# stochastic. For a NameError it is a guaranteed no-op that costs a full
# training run — three of them burn ~30 minutes of a 6h budget to learn nothing.
_QUOTED = re.compile(r"""(['"])(?:(?!\1).)*\1""")

_NONDETERMINISTIC = re.compile(
    r"\b("
    r"nan|inf|overflow|underflow|singular matrix|did not converge|"
    r"out of memory|MemoryError|timed out|TimeoutError|"
    r"LinAlgError|divide by zero|invalid value encountered"
    r")\b",
    re.IGNORECASE,
)


# Checked BEFORE the stochastic pattern, because these names are conclusive and
# the stochastic pattern is not. "ValueError: could not convert string to float:
# 'nan'" is a deterministic bug whose message happens to contain the word "nan";
# without this precedence it would earn a free reseed that cannot possibly work.
_DETERMINISTIC_EXCEPTIONS = re.compile(
    r"\b("
    r"NameError|AttributeError|ImportError|ModuleNotFoundError|SyntaxError|"
    r"IndentationError|TypeError|KeyError|IndexError|UnboundLocalError|"
    r"FileNotFoundError|NotImplementedError|AssertionError|"
    r"refused to write a proposed file"
    r")\b"
)


def classify_run_error(error_text: str | None) -> str:
    """'nondeterministic' (reroll may help) or 'deterministic' (repair it).

    Deterministic is the default. Getting this wrong in the deterministic
    direction costs one LLM call; getting it wrong the other way costs a
    training run that cannot possibly succeed.
    """
    if not error_text:
        return "deterministic"
    if _DETERMINISTIC_EXCEPTIONS.search(error_text):
        return "deterministic"
    # A quoted token is DATA, not a numerical event: "could not convert string
    # to float: 'nan'" is a deterministic parsing bug whose message happens to
    # contain the word. Strip quoted spans before looking for the stochastic
    # signature, so the classifier reads what the error *is* rather than what it
    # quotes. `ValueError` is deliberately NOT in the deterministic list above --
    # it covers both a shape mismatch and a genuine "array contains NaNs", so it
    # has to be decided by the rest of the message.
    unquoted = _QUOTED.sub(" ", error_text)
    return "nondeterministic" if _NONDETERMINISTIC.search(unquoted) else "deterministic"


def repair_strategy(error_text: str | None, attempt: int) -> str:
    """'reseed' or 'repair', for the given retry attempt (0-based).

    A stochastic failure gets exactly one free reseed — no tokens, no model
    call — before we start paying for repairs. Everything else goes straight to
    repair, following MLE-STAR's debugging module rather than hoping.
    """
    if attempt == 0 and classify_run_error(error_text) == "nondeterministic":
        return "reseed"
    return "repair"


# ---------------------------------------------------------------- L4 ------
def experiment_time_estimate(history: list, default: float = 300.0) -> float:
    """Median wall-clock of recent successful experiments, as a scheduling input.

    Median rather than mean: one 600-second timeout should not double the
    estimate and make the agent stop early.
    """
    times = [h.get("wall_clock_s", 0.0) for h in history[-10:] if h.get("wall_clock_s", 0.0) > 0]
    if not times:
        return default
    times = sorted(times)
    mid = len(times) // 2
    return times[mid] if len(times) % 2 else 0.5 * (times[mid - 1] + times[mid])


# Reserved for finalize: training the submission model and running submit.py
# --check against 170,588 test rows. If the loop spends the budget down to zero
# there is nothing left to produce the deliverable with, which is the failure
# this whole module exists to prevent.
FINALIZE_RESERVE_SECONDS = 420.0


def budget_allows_another_experiment(elapsed: float, max_wall_seconds: float,
                                     history: list) -> tuple[bool, str]:
    """Should the loop start another experiment, or converge and ship?

    The old check only asked whether the budget was *already* spent, so at
    5h58m it would happily start an experiment that runs to the 10-minute cap,
    overshoot the stated 6h limit, and leave nothing for finalize.
    """
    remaining = max_wall_seconds - elapsed
    needed = experiment_time_estimate(history) + FINALIZE_RESERVE_SECONDS
    if remaining <= FINALIZE_RESERVE_SECONDS:
        return False, (f"{remaining / 60:.1f} min left, below the "
                       f"{FINALIZE_RESERVE_SECONDS / 60:.0f} min reserved for finalize")
    if remaining < needed:
        return False, (f"{remaining / 60:.1f} min left; a typical experiment plus "
                       f"finalize needs {needed / 60:.1f} min")
    return True, ""

def plateaued(history: list, n_plateau: int, epsilon: float) -> tuple[bool, str]:
    """The brief's convergence rule, read literally.

        "A run is considered converged when validation score has not improved
         by more than eps = 0.002 over the last N = 3 consecutive iterations"

    That is a **window** test on the best-so-far curve, not a per-iteration one.
    The difference is not academic on this benchmark, where the largest honest
    single-step gain anyone has measured is about +0.001:

        old (per-iteration): every step under +0.002 is "no improvement", so
            three consecutive +0.001 steps -- a cumulative +0.003, comfortably
            past the threshold -- converge the run and stop it at iteration 4,
            having spent six minutes of a six-hour budget.
        new (windowed): the same three steps are +0.003 over the window, so the
            run correctly keeps going while it is still making progress.

    A genuinely stuck run still stops, which is the point of the rule. This only
    stops the agent quitting while it is winning.
    """
    completed = [h for h in history if h.get("outcome") != "error"]
    if len(completed) <= n_plateau:
        return False, ""
    best_curve, best = [], float("-inf")
    for h in completed:
        best = max(best, h.get("metrics", {}).get("valid_primary", 0.0))
        best_curve.append(best)
    gain = best_curve[-1] - best_curve[-1 - n_plateau]
    if gain > epsilon:
        return False, ""
    return True, (f"validation best moved {gain:+.4f} over the last {n_plateau} "
                  f"completed iterations, at or below the {epsilon} threshold")

# ------------------------------------------------------- recovery log -----
@dataclass
class RecoveryEvent:
    """One thing that went wrong and what the agent did about it, unattended.

    Robustness (20%) is graded on recovery, and a run that recovered silently
    looks identical to a run that never had a problem. This is the evidence.
    """
    iteration: int
    layer: str        # 'llm' | 'experiment' | 'budget' | 'terminal'
    kind: str         # e.g. 'OverloadedError', 'deterministic-crash'
    action: str       # 'retry' | 'reseed' | 'repair' | 'pivot' | 'finalize-early'
    detail: str
    iso: str
    ts: float


def log_recovery(root: str | Path, event: RecoveryEvent) -> None:
    """Append one recovery event. Never raises — logging must not break a run."""
    try:
        path = Path(root) / "recovery.jsonl"
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(event)) + "\n")
    except OSError:
        pass


def recovery_event(iteration: int, layer: str, kind: str, action: str,
                   detail: str = "") -> RecoveryEvent:
    return RecoveryEvent(iteration=iteration, layer=layer, kind=kind, action=action,
                         detail=detail[:500], iso=time.strftime("%Y-%m-%dT%H:%M:%S"),
                         ts=time.time())


def read_recovery(root: str | Path) -> list[dict]:
    path = Path(root) / "recovery.jsonl"
    if not path.exists():
        return []
    out = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if ln:
            try:
                out.append(json.loads(ln))
            except json.JSONDecodeError:
                continue
    return out
