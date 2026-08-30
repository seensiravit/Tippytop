"""Feature engineering (STUB).

Reminder from the measured ablations: adding *static* fields and pure user-side
features does NOT move the score (ranking is within-user). Payoff is in signals
that VARY within a user — watch time, engagement labels, time-of-day — used via
crosses with item-side features. Build those here.
"""
from __future__ import annotations

# TODO: extend the kit's row tuple / FIELDS with within-user-varying features.
