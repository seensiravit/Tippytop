"""Our data layer — thin wrappers over the kit's frozen ``load``/``encode``.

Feature engineering that stays WITHIN the kit's contract goes in ``features.py``;
per-user history construction goes in ``sequences.py``. The kit's own
``data.load``/``data.encode`` are re-exported via ``tippytop.kit``.
"""
from ..kit import load, encode, FIELDS, LABEL, SPLITS

__all__ = ["load", "encode", "FIELDS", "LABEL", "SPLITS"]
