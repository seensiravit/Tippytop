"""Run-integrity utilities shared by the agent harness and the graders' artifacts.

These two modules used to live inside ``src/tippytop/agent/`` — the earlier,
now-deleted Gemini agent. They have nothing to do with *which* agent runs:

* ``interventions`` measures the manual-intervention count, which Deliverable 3
  requires and which Impact & Relevance (20%) is scored on.
* ``redact`` scrubs hidden-test signal out of any text on its way into an LLM
  prompt, which is what keeps the walled-validation claim true rather than
  merely asserted.

Deleting the old agent wholesale would have deleted both. They were moved here
first, and are now wired into ``autoresearch_lg`` where they actually belong.
"""
from .interventions import Intervention, InterventionLog, KINDS
from .redact import scrub, contains_test_signal, REDACTION

__all__ = ["Intervention", "InterventionLog", "KINDS",
           "scrub", "contains_test_signal", "REDACTION"]
