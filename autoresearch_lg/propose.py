"""The propose sub-graph — the brain (Innovation + Autonomy).

    read_mode -> build_context -> retrieve_options -> llm_generate -> valid?
                                                              ^              \\
                                                              |__ regenerate  \\
                                                                  (once)       -> END

`read_mode` takes the tune/expand/pivot mode the router already set on
state — it does not decide mode itself (mode vs. content is the router's
job vs. this sub-graph's, kept as two separate authorities so the LLM can't
quietly override the router's macro/micro decision).

`llm_generate` is where a NEW concept gets opened (mode in expand/pivot) or
the ACTIVE one continues (mode=tune) — see _open_or_continue_concept.
`validate_diff` does an actual `ast.parse()` syntax check (not just trusting
the model's own say-so) and regenerates once if the proposed code doesn't
parse; if it's still broken after that, it's let through anyway — the
experiment sub-graph's own failure handling will catch it as an 'error'
outcome, which is a legitimate signal in its own right (a genuinely
malformed idea, not a harness bug).
"""
from __future__ import annotations

import ast
import json

import anthropic
import openai
from langgraph.graph import END, StateGraph

from . import bootstrap
from . import context as context_mod
from . import resilience, tools
from .state import ResearchState

DEFAULT_MODEL = bootstrap.DEFAULT_MODEL  # single source of truth — bootstrap.py

# Models that support adaptive thinking + output_config.effort. Older models
# (Haiku 4.5, Sonnet 4.5, ...) 400 on these params — omit them entirely there
# rather than trying to translate to the legacy budget_tokens form, since
# swapping to a cheap model is for quick pipeline smoke-tests, not thinking
# quality.
_ADAPTIVE_THINKING_MODELS = {
    "claude-opus-5", "claude-opus-4-8", "claude-opus-4-7", "claude-opus-4-6",
    "claude-sonnet-5", "claude-sonnet-4-6", "claude-fable-5", "claude-mythos-5",
}


def _anthropic_request_kwargs(model: str) -> dict:
    if model in _ADAPTIVE_THINKING_MODELS:
        return {"thinking": {"type": "adaptive"}, "output_config": {"effort": "high"}}
    return {}


def _is_anthropic_model(model: str) -> bool:
    return model.startswith("claude-")


# Shared tool definition — Anthropic and OpenAI wrap the same name/
# description/JSON-schema in different envelopes (see PROPOSE_TOOL vs
# OPENAI_TOOL below), so it's defined once here to avoid the two drifting
# apart as the schema changes.
_TOOL_NAME = "propose_experiment"
_TOOL_DESCRIPTION = (
    "Generate the next experiment: a hypothesis shaped by the given mode, "
    "and the complete new contents of every file you're changing (full "
    "file, not a diff) to run it."
)
_TOOL_SCHEMA = {
    "type": "object",
    "properties": {
        "concept": {
            "type": "string",
            "description": (
                "The concept name. mode=tune: repeat the active concept's "
                "name verbatim. mode=expand/pivot: a new, short concept name, "
                "e.g. 'listwise softmax loss'."
            ),
        },
        "hypothesis": {
            "type": "string",
            "description": "The falsifiable claim this experiment tests, grounded in the EDA and context given.",
        },
        "description": {
            "type": "string",
            "description": "Short summary of THIS SPECIFIC run for the log, e.g. 'softmax temp=0.5, lr=0.001' — distinct from the concept name, which persists across tune retries.",
        },
        "files": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "enum": ["baseline.py", "data.py"]},
                    "content": {"type": "string", "description": "Complete new file content."},
                },
                "required": ["path", "content"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["concept", "hypothesis", "description", "files"],
    "additionalProperties": False,
}

# Anthropic Messages API tool shape.
PROPOSE_TOOL = {
    "name": _TOOL_NAME,
    "description": _TOOL_DESCRIPTION,
    "input_schema": _TOOL_SCHEMA,
    "strict": True,
}

# OpenAI Chat Completions tool shape — verified against the installed SDK
# (openai==3.6.0 here; ChatCompletionToolParam / FunctionDefinition /
# ChatCompletionMessage.tool_calls[i].function.arguments) rather than
# assumed, since the SDK has moved fast and stale-memory API shapes are a
# real risk. Re-verify against `openai.types.chat.*` if this ever 400s.
OPENAI_TOOL = {
    "type": "function",
    "function": {
        "name": _TOOL_NAME,
        "description": _TOOL_DESCRIPTION,
        "parameters": _TOOL_SCHEMA,
        "strict": True,
    },
}


def _call_anthropic(model: str, system_prompt: str, user_content: str) -> tuple[dict, int, int]:
    # The SDK default is max_retries=2 with its own backoff. That is tuned for
    # an interactive call; this one runs unattended for hours, where the cost of
    # giving up is the entire run. See resilience.CLIENT_MAX_RETRIES.
    client = anthropic.Anthropic(max_retries=resilience.CLIENT_MAX_RETRIES,
                                 timeout=resilience.CLIENT_TIMEOUT_SECONDS)
    response = client.messages.create(
        model=model,
        max_tokens=16000,
        system=[{
            "type": "text",
            "text": system_prompt,
            "cache_control": {"type": "ephemeral"},
        }],
        tools=[PROPOSE_TOOL],
        tool_choice={"type": "tool", "name": _TOOL_NAME},
        messages=[{"role": "user", "content": user_content}],
        **_anthropic_request_kwargs(model),
    )
    # A bare next() raises StopIteration here, which generator machinery
    # upstream can swallow or reinterpret — the run then dies with a traceback
    # that points nowhere near the real cause. This happens for real: the model
    # can answer with text only when it hits max_tokens mid-tool-use, or when it
    # declines. Raise something the retry policy can recognise instead.
    tool_use = next((b for b in response.content if b.type == "tool_use"), None)
    if tool_use is None:
        kinds = [getattr(b, "type", "?") for b in response.content]
        raise resilience.ProposalError(
            f"no tool_use block in the response (stop_reason="
            f"{getattr(response, 'stop_reason', '?')}, blocks={kinds})")
    return _validate_payload(tool_use.input), response.usage.input_tokens, response.usage.output_tokens


def _call_openai(model: str, system_prompt: str, user_content: str) -> tuple[dict, int, int]:
    client = openai.OpenAI(max_retries=resilience.CLIENT_MAX_RETRIES,
                           timeout=resilience.CLIENT_TIMEOUT_SECONDS)
    response = client.chat.completions.create(
        model=model,
        max_completion_tokens=16000,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        tools=[OPENAI_TOOL],
        tool_choice={"type": "function", "function": {"name": _TOOL_NAME}},
    )
    calls = (response.choices[0].message.tool_calls or []) if response.choices else []
    if not calls:
        raise resilience.ProposalError(
            f"no tool call in the response (finish_reason="
            f"{response.choices[0].finish_reason if response.choices else '?'})")
    try:
        payload = json.loads(calls[0].function.arguments)
    except json.JSONDecodeError as e:
        raise resilience.ProposalError(f"tool arguments were not valid JSON: {e}") from e
    return (_validate_payload(payload),
            response.usage.prompt_tokens, response.usage.completion_tokens)


_REQUIRED_KEYS = ("concept", "hypothesis", "description", "files")


def _validate_payload(payload: dict) -> dict:
    """Fail here, with a readable reason, rather than at a KeyError three
    functions away. `strict` schemas make this rare; rare is not never, and a
    six-hour unattended run is exactly where the rare case shows up."""
    if not isinstance(payload, dict):
        raise resilience.ProposalError(f"tool payload is {type(payload).__name__}, not an object")
    missing = [k for k in _REQUIRED_KEYS if not payload.get(k)]
    if missing:
        raise resilience.ProposalError(f"tool payload is missing {missing}")
    files = payload["files"]
    if not isinstance(files, list) or not files:
        raise resilience.ProposalError("tool payload has no files")
    seen = set()
    for f in files:
        if not isinstance(f, dict) or "path" not in f or "content" not in f:
            raise resilience.ProposalError(f"malformed file entry: {str(f)[:200]}")
        # `{f["path"]: f["content"] for f in files}` silently keeps the LAST
        # entry for a repeated path. That is deterministic but it is not
        # obviously the model's intent, and a proposal that describes a file
        # twice is a proposal we do not understand. Regenerating costs one call;
        # guessing costs a wasted training run on code nobody chose.
        if f["path"] in seen:
            raise resilience.ProposalError(
                f"file {f['path']!r} appears more than once; one entry per file")
        seen.add(f["path"])
    return payload


def read_mode(state: ResearchState) -> dict:
    # Pass-through node: mode is already set on state by router() in graph.py.
    # Exists as its own node so Studio/mermaid show the pipeline's real shape
    # (technical-plan.md's sub-graph diagram), and resets per-call scratch.
    return {"propose_attempt": 0, "diff_error": "", "tokens_in": 0, "tokens_out": 0}


def build_context(state: ResearchState) -> dict:
    return {"context_summary": context_mod.build_context(state)}


def retrieve_options(state: ResearchState) -> dict:
    return {"retrieved_options": context_mod.retrieve_options(state)}


def _mode_instructions(state: ResearchState) -> str:
    mode = state["mode"]
    active = next((c for c in state["concepts"] if c["id"] == state["active_concept_id"]), None)
    if not active and not state["concepts"]:
        return (
            "MODE: first proposal. No concept has been tried yet beyond the "
            "baseline. Propose the strongest first concept given the EDA and "
            "the available directions below."
        )
    if mode == "repair":
        # MLE-STAR's debugging module: hand back the traceback and ask for a
        # fix, keeping the concept fixed. A blind reroll at another seed cannot
        # fix a NameError, and spends a full training run finding that out.
        return (
            f"MODE: repair (debug). Your last experiment for the ACTIVE concept "
            f"'{active['statement'] if active else '?'}' FAILED TO RUN. Below is "
            "the code exactly as it was executed, and the error it produced.\n\n"
            "Fix the defect. Do NOT change the concept, do NOT propose a "
            "different idea, and do NOT simplify the idea away to make the error "
            "go away — a repair that removes the hypothesis is worth nothing. "
            "`concept` must match the active concept verbatim.\n\n"
            "The harness runs ONLY `baseline.py --model fm`, so verify your fix "
            "is reachable from `run_fm`.\n\n"
            f"ERROR:\n{state.get('repair_error', '(no detail captured)')[:3000]}"
        )
    if mode == "tune":
        return (
            f"MODE: tune (micro exploit). Continue the ACTIVE concept "
            f"'{active['statement'] if active else '?'}' — same underlying "
            "idea, different knobs (hyperparameters, a bugfix, a small "
            "variant). concept must match it verbatim. Log line should read "
            "as: tuning concept X further after an improvement."
        )
    if mode == "expand":
        return (
            f"MODE: expand (macro exploit). The concept "
            f"'{active['statement'] if active else '(just closed)'}' worked "
            "and has been tuned enough — propose a NEW but ADJACENT concept "
            "that builds on what just worked, not a random departure. "
            "This is exploiting a success, not fleeing a failure."
        )
    return (  # pivot
        "MODE: pivot (macro explore). The active concept failed or is "
        "unrecoverably broken — propose a genuinely NEW concept, not a "
        "variant of what just failed. This is exploring away from a dead "
        "end, not a tweak."
    )


def llm_generate(state: ResearchState) -> dict:
    # "Current" = the best experiment's own folder if one exists, else the
    # pristine root files (nothing has beaten the baseline yet). Never the
    # root files once any concept has actually been kept — root baseline.py/
    # data.py are read-only from this harness's point of view after setup.
    # In repair mode the model must see the code that actually crashed, not the
    # last good code — otherwise it is debugging a file it never wrote.
    if state["mode"] == "repair" and state.get("repair_exp_dir"):
        source_dir = state["repair_exp_dir"]
    else:
        source_dir = state.get("best_exp_dir") or state["repo_root"]
    fallback_note = ""
    try:
        current_files = tools.read_experiment_files(source_dir, state["editable_files"])
    except OSError as e:
        # The incumbent's folder is gone (deleted, a full disk, a cleaned runs/).
        # This is a LOCAL, permanent failure, and without this it surfaces as a
        # bare OSError inside a node whose retry policy would back off five times
        # and then declare the provider dead -- the wrong diagnosis and the wrong
        # response. Fall back to the pristine root files, which always exist, and
        # say so rather than silently regressing to the baseline.
        resilience.log_recovery(state["repo_root"], resilience.recovery_event(
            iteration=state.get("iteration", 0), layer="experiment",
            kind="missing-source-dir", action="fallback-to-root",
            detail=f"{source_dir}: {e}"))
        source_dir = state["repo_root"]
        current_files = tools.read_experiment_files(source_dir, state["editable_files"])
        fallback_note = (
            "\nNOTE: the incumbent experiment folder could not be read, so the "
            "files below are the pristine baseline, NOT the current best. Treat "
            "this as a fresh start from the baseline.")
    parts = [
        _mode_instructions(state),
        fallback_note + "\nCurrent file contents:\n" + "\n\n".join(
            f"### {p}\n```python\n{c}\n```" for p, c in current_files.items()
        ),
        "\nEDA on the current train/valid splits:\n" + _eda_block(state["eda_summary"]),
        "\nRun history:\n" + state["context_summary"],
        "\n" + state["retrieved_options"],
    ]
    # A bit-identical result means the harness ran code the model did not
    # change -- overwhelmingly, a new --model choice added while run_fm, the
    # only path the harness invokes, was left untouched. Say so loudly: without
    # it the model reads "no change" as "my idea is neutral" and tunes something
    # that never runs. Cost two iterations on run3 before it was caught.
    history = state.get("history") or []
    if history and abs(history[-1]["metrics"]["valid_primary"]
                       - state["best_valid_primary"]) < 1e-9:
        parts.append(
            "\nWARNING: the previous experiment scored EXACTLY the incumbent, to "
            "four decimals. That is almost never a real result — it means the "
            "harness executed code you did not change. The harness runs ONLY "
            "`baseline.py --model fm`, so an idea added as a new --model choice, "
            "or anywhere outside the `run_fm` path, never executes. Check that "
            "your change is reachable from run_fm before proposing anything else."
        )
    if state["diff_error"]:
        parts.append(
            f"\nYour previous proposal this round did not parse as valid "
            f"Python and is being regenerated: {state['diff_error']}\n"
            "Fix the syntax error — keep the same concept and hypothesis."
        )
    parts.append("\nCall propose_experiment now.")
    user_content = "\n".join(parts)

    model = state.get("model") or DEFAULT_MODEL
    call = _call_anthropic if _is_anthropic_model(model) else _call_openai
    payload, tokens_in, tokens_out = call(model, state["system_prompt"], user_content)

    edited = {f["path"]: f["content"] for f in payload["files"]}
    concepts, active_id = _open_or_continue_concept(state, payload["concept"], payload["hypothesis"])

    return {
        "idea_concept": payload["concept"],
        "idea_hypothesis": payload["hypothesis"],
        "idea_description": payload["description"],
        "edited_files": edited,
        "concepts": concepts,
        "active_concept_id": active_id,
        "propose_attempt": state["propose_attempt"] + 1,
        "tokens_in": state.get("tokens_in", 0) + tokens_in,
        "tokens_out": state.get("tokens_out", 0) + tokens_out,
    }


def _open_or_continue_concept(state: ResearchState, statement: str, rationale: str) -> tuple[list, str]:
    concepts = [dict(c) for c in state["concepts"]]
    if state["mode"] == "tune" and state["active_concept_id"]:
        return concepts, state["active_concept_id"]  # continue existing, unchanged
    new_id = f"c{len(concepts) + 1}"
    concepts.append({
        "id": new_id,
        "statement": statement,
        "rationale": rationale,
        "status": "active",
        "closed_reason": "",
        "opened_at_iteration": state["iteration"],
        "attempts": [],
    })
    return concepts, new_id


def _eda_block(eda: dict) -> str:
    if not eda:
        return "(not computed)"
    if "error" in eda:
        return f"EDA failed (non-fatal): {eda['error']}"
    return json.dumps(eda, indent=2)


# AIDE's published debug budget (aide/utils/config.yaml: search.max_debug_depth
# = 3). Anchored to a value that has been run at scale rather than guessed; two
# was too few to be worth the round trip, and more than three on a *syntax*
# defect means the model is not going to get it.
MAX_REGENERATE_ATTEMPTS = 3


def validate_diff(state: ResearchState) -> dict:
    errors = []
    for path, content in state["edited_files"].items():
        try:
            ast.parse(content, filename=path)
        except SyntaxError as e:
            errors.append(f"{path}: {e}")
    return {"diff_valid": not errors, "diff_error": "; ".join(errors)}


def _route_validate(state: ResearchState) -> str:
    if state["diff_valid"] or state["propose_attempt"] >= MAX_REGENERATE_ATTEMPTS:
        # valid, or out of regeneration attempts — let it through either way; a
        # still-broken diff becomes a legitimate 'error' outcome downstream,
        # which the router then routes into a repair rather than a reroll.
        return "ok"
    return "regenerate"


def build_propose_graph():
    g = StateGraph(ResearchState)
    g.add_node("read_mode", read_mode)
    g.add_node("build_context", build_context)
    g.add_node("retrieve_options", retrieve_options)
    g.add_node("llm_generate", llm_generate)
    g.add_node("validate_diff", validate_diff)

    g.set_entry_point("read_mode")
    g.add_edge("read_mode", "build_context")
    g.add_edge("build_context", "retrieve_options")
    g.add_edge("retrieve_options", "llm_generate")
    g.add_edge("llm_generate", "validate_diff")
    g.add_conditional_edges(
        "validate_diff", _route_validate,
        {"regenerate": "llm_generate", "ok": END},
    )
    return g.compile()
