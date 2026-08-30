"""Experiment orchestration that wraps training rather than replacing it."""
from .block import Candidate, BlockResult, ArmResult, run_block, plan_block

__all__ = ["Candidate", "BlockResult", "ArmResult", "run_block", "plan_block"]
