"""Trusted construction of label-safe data frames for generated experiments."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


SPLIT_DATES = {
    "train": (20220408, 20220421),
    "valid": (20220422, 20220428),
    "test": (20220429, 20220508),
}
PREDICTION_COLUMNS = [
    "date",
    "user_id",
    "video_id",
    "author_id",
    "tab",
    "duration_ms",
    "hourmin",
    "time_ms",
]
AUXILIARY_COLUMNS = [
    "play_time_ms",
    "is_click",
    "is_like",
    "is_follow",
    "is_comment",
    "is_forward",
    "is_hate",
    "profile_stay_time",
    "comment_stay_time",
    "is_profile_enter",
]
TRAINING_COLUMNS = [*PREDICTION_COLUMNS, "long_view", *AUXILIARY_COLUMNS]
_LOG_FILES = {
    "train": "log_standard_4_08_to_4_21_pure.csv",
    "valid": "log_standard_4_22_to_5_08_pure.csv",
    "test": "log_standard_4_22_to_5_08_pure.csv",
}


def load_research_frames(data_dir: Path) -> dict[str, pd.DataFrame]:
    """Return labeled train and feature-only validation frames in benchmark order."""

    authors = _load_authors(data_dir)
    train = _load_split(data_dir, "train", authors, training=True)
    valid = _load_split(data_dir, "valid", authors, training=False)
    return {"train": train, "valid": valid}


def load_prediction_frame(data_dir: Path, split: str) -> pd.DataFrame:
    if split not in SPLIT_DATES:
        raise ValueError(f"unknown split: {split!r}")
    return _load_split(data_dir, split, _load_authors(data_dir), training=False)


def prediction_view(frame: pd.DataFrame) -> pd.DataFrame:
    """Strip every outcome column, including auxiliary training-only outcomes."""

    missing = [column for column in PREDICTION_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"research frame is missing prediction columns: {missing}")
    return frame.loc[:, PREDICTION_COLUMNS].copy()


def _load_authors(data_dir: Path) -> dict[str, str]:
    path = data_dir / "video_features_basic_pure.csv"
    frame = pd.read_csv(path, usecols=["video_id", "author_id"], dtype=str)
    return dict(zip(frame["video_id"], frame["author_id"], strict=True))


def _load_split(
    data_dir: Path,
    split: str,
    authors: dict[str, str],
    *,
    training: bool,
) -> pd.DataFrame:
    requested = [
        "date",
        "user_id",
        "video_id",
        "tab",
        "duration_ms",
        "hourmin",
        "time_ms",
    ]
    if training:
        requested.extend(["long_view", *AUXILIARY_COLUMNS])
    path = data_dir / _LOG_FILES[split]
    available = pd.read_csv(path, nrows=0).columns
    usecols = [column for column in requested if column in available]
    frame = pd.read_csv(
        path,
        usecols=usecols,
        dtype={"user_id": str, "video_id": str, "tab": str},
    )
    lo, hi = SPLIT_DATES[split]
    frame = frame.loc[frame["date"].between(lo, hi)].reset_index(drop=True)
    frame["author_id"] = frame["video_id"].map(authors).fillna("UNK")

    # Small synthetic fixtures may omit optional context/outcome columns.
    for column in PREDICTION_COLUMNS:
        if column not in frame:
            frame[column] = 0
    if training:
        for column in ["long_view", *AUXILIARY_COLUMNS]:
            if column not in frame:
                frame[column] = 0
        frame["long_view"] = (pd.to_numeric(frame["long_view"], errors="coerce").fillna(0) != 0).astype("int8")
        return frame.loc[:, TRAINING_COLUMNS]
    return prediction_view(frame)
