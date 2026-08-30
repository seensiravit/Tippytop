"""Central constants and paths. Import from here instead of hard-coding."""
from .kit import REPO_ROOT, DEFAULT_DATA_DIR, SPLITS, LABEL

DATA_DIR = DEFAULT_DATA_DIR          # override via CLI/env when data lives elsewhere
RESULTS_DIR = REPO_ROOT / "results"
SUBMISSIONS_DIR = REPO_ROOT / "results" / "submissions"

# Baselines to beat (from baseline_scores.json / README).
FM_BASELINE_PRIMARY = 0.5946
ORACLE_CEILING_PRIMARY = 0.8645
RANDOM_SANITY_PRIMARY = 0.4753        # --model random must reproduce ~0.475

# Convergence: valid primary gain <= EPS for N consecutive iters => converged.
CONVERGENCE_EPS = 0.002
CONVERGENCE_N = 3

DEFAULT_SEED = 42

__all__ = [
    "REPO_ROOT", "DATA_DIR", "RESULTS_DIR", "SUBMISSIONS_DIR", "SPLITS", "LABEL",
    "FM_BASELINE_PRIMARY", "ORACLE_CEILING_PRIMARY", "RANDOM_SANITY_PRIMARY",
    "CONVERGENCE_EPS", "CONVERGENCE_N", "DEFAULT_SEED",
]
