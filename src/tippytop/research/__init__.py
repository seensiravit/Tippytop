"""Label-safe APIs and schemas for autonomous ML research."""

from .api import CategoricalEncoder, TabularEncoder, labels, user_ids
from .plan import ResearchPlan

__all__ = [
    "CategoricalEncoder",
    "ResearchPlan",
    "TabularEncoder",
    "labels",
    "user_ids",
]
