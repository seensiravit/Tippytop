"""One bundle that carries everything a model needs, built once per run.

``splits``  : raw rows per partition, from the kit's ``load`` (row tuple is
              ``(date, user_id, video_id, author_id, tab, duration_ms, long_view)``).
``enc``     : encoded ``(X, y, users)`` per partition, from the kit's ``encode``.
``dim``     : total shared-embedding-table size.

Row order is identical between ``splits[name]`` and ``enc[name]`` (encode
preserves order), so a model may use whichever view it needs and the scores line
up with both — and with the submission's ``row_id``.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from ..kit import load, encode


@dataclass
class Dataset:
    splits: dict            # name -> list of raw row tuples
    enc: dict               # name -> (X int32 (N,F), y float32 (N,), users list)
    dim: int

    def rows(self, split: str) -> list:
        return self.splits[split]

    def X(self, split: str) -> np.ndarray:
        return self.enc[split][0]

    def y(self, split: str) -> np.ndarray:
        return self.enc[split][1]

    def users(self, split: str) -> list:
        return self.enc[split][2]


def load_dataset(data_dir) -> Dataset:
    """Load + encode once. ``data_dir`` may be a str or Path."""
    splits = load(str(data_dir))
    enc, dim = encode(splits)
    return Dataset(splits=splits, enc=enc, dim=dim)
