"""Statistical infrastructure — measure the instrument before trusting a comparison.

Nothing in this package trains a model. Every module here quantifies uncertainty
in numbers that already exist, which is why it is the cheapest rigor available:

``bootstrap``  paired CIs over users, no retraining at all
``power``      how many seeds an effect needs, and what selection noise inflates
``transfer``   the validation->test slope every plan has been assuming
``selection``  submit a top-K rank-average rather than a validation argmax
"""
from .bootstrap import (per_user_stats, primary_from, paired_bootstrap, diagnose_eval_split,
                        bootstrap_ci, verify_against_kit, format_comparison,
                        PerUserStats)
from .power import (seeds_needed, is_detectable, expected_max, selection_inflation,
                    recommend_seeds, EPSILON, DEFAULT_SIGMA_VALID, DEFAULT_SIGMA_TEST)
from .transfer import transfer_slope, rescale_target, required_valid_gain
from .selection import (rank_within_user, rank_average, topk_selection,
                        compare_argmax_vs_topk, SelectionResult)

__all__ = [
    "per_user_stats", "primary_from", "paired_bootstrap", "bootstrap_ci",
    "verify_against_kit", "format_comparison", "PerUserStats", "diagnose_eval_split",
    "seeds_needed", "is_detectable", "expected_max", "selection_inflation",
    "recommend_seeds", "EPSILON", "DEFAULT_SIGMA_VALID", "DEFAULT_SIGMA_TEST",
    "transfer_slope", "rescale_target", "required_valid_gain",
    "rank_within_user", "rank_average", "topk_selection",
    "compare_argmax_vs_topk", "SelectionResult",
]
