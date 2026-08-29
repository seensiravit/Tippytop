"""Autonomous ML research agent — AIDE-style code-optimization loop.

The agent iteratively writes a full runnable solution.py (via an LLM), runs it in
a sandbox on the valid split, keeps the best-by-valid, and touches test only once
at finalize. See docs/agent.md.
"""
from .config import AgentConfig
from .orchestrator import run_agent, AgentState
from .llm import build_llm_client, BaseLLMClient, MockLLMClient, GeminiClient

__all__ = ["AgentConfig", "run_agent", "AgentState", "build_llm_client",
           "BaseLLMClient", "MockLLMClient", "GeminiClient"]
