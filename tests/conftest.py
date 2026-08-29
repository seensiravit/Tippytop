from __future__ import annotations

import csv
from pathlib import Path

import pytest


def row(date: int, user: str, video: str, author: str, label: int) -> tuple[object, ...]:
    return (date, user, video, author, "1", 1000.0, label)


@pytest.fixture
def synthetic_splits() -> dict[str, list[tuple[object, ...]]]:
    train = []
    for _ in range(5):
        train.extend(
            [
                row(20220408, "u1", "v1", "a1", 1),
                row(20220408, "u1", "v2", "a2", 0),
                row(20220408, "u2", "v3", "a3", 1),
                row(20220408, "u2", "v4", "a4", 0),
            ]
        )
    valid = [
        row(20220422, "u1", "v1", "a1", 1),
        row(20220422, "u1", "v2", "a2", 0),
        row(20220422, "u2", "v3", "a3", 1),
        row(20220422, "u2", "v4", "a4", 0),
    ]
    test = [
        row(20220429, "u1", "v1", "a1", 1),
        row(20220429, "u1", "v2", "a2", 0),
        row(20220429, "u2", "v3", "a3", 1),
        row(20220429, "u2", "v4", "a4", 0),
    ]
    return {"train": train, "valid": valid, "test": test}


@pytest.fixture
def synthetic_data_dir(tmp_path: Path) -> Path:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    with (data_dir / "video_features_basic_pure.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["video_id", "author_id"])
        writer.writerows([[f"v{index}", f"a{index}"] for index in range(1, 5)])

    header = ["date", "user_id", "video_id", "tab", "duration_ms", "long_view"]
    first_rows = []
    for _ in range(5):
        first_rows.extend(
            [
                [20220408, "u1", "v1", 1, 1000, 1],
                [20220408, "u1", "v2", 1, 1000, 0],
                [20220408, "u2", "v3", 1, 1000, 1],
                [20220408, "u2", "v4", 1, 1000, 0],
            ]
        )
    second_rows = [
        [20220422, "u1", "v1", 1, 1000, 1],
        [20220422, "u1", "v2", 1, 1000, 0],
        [20220422, "u2", "v3", 1, 1000, 1],
        [20220422, "u2", "v4", 1, 1000, 0],
        [20220429, "u1", "v1", 1, 1000, 1],
        [20220429, "u1", "v2", 1, 1000, 0],
        [20220429, "u2", "v3", 1, 1000, 1],
        [20220429, "u2", "v4", 1, 1000, 0],
    ]
    for filename, rows in (
        ("log_standard_4_08_to_4_21_pure.csv", first_rows),
        ("log_standard_4_22_to_5_08_pure.csv", second_rows),
    ):
        with (data_dir / filename).open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            writer.writerows(rows)
    return data_dir
