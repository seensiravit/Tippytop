"""Per-user behaviour sequences (STUB).

Each user has hundreds–thousands of train interactions, entirely unused by the
baseline. Build ordered per-user history here to feed DIN/SIM-style models.
"""
from __future__ import annotations

# TODO: group rows by user, order by time_ms/hourmin, expose fixed-length or
# variable-length history windows keyed to each eval impression (no leakage:
# only history strictly before the impression being scored).
