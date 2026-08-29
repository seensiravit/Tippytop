"""Generate a tiny KuaiRand-shaped dataset so agent tests run in milliseconds.

Writes the three CSVs the frozen kit's data.load() reads, with just enough
structure that the FM trains and evaluate() produces a non-degenerate primary.
"""
from __future__ import annotations
from pathlib import Path
import csv

N_USERS = 40
N_VIDEOS = 15
N_AUTHORS = 6
_DUR = [3000, 5000, 8000, 12000, 15000, 20000, 25000, 30000,
        40000, 6000, 9000, 11000, 18000, 22000, 35000]

# date buckets inside each official split window
_TRAIN_DATE, _VALID_DATE, _TEST_DATE = 20220410, 20220424, 20220502

_LOG_HEADER = ["date", "user_id", "video_id", "tab", "duration_ms",
               "play_time_ms", "is_click", "long_view"]


def _label(u: int, v: int) -> int:
    # structured so FM can learn a user x video signal; ~per-user variation
    return 1 if ((u * 7 + v * 13) % 5) < 2 else 0


def _rows_for(date: int, videos_per_user: int, offset: int):
    rows = []
    for u in range(N_USERS):
        for j in range(videos_per_user):
            v = (u * 3 + j * 5 + offset) % N_VIDEOS
            lab = _label(u, v)
            dur = _DUR[v]
            play = int(dur * (0.8 if lab else 0.2))
            rows.append([date, u, v, u % 4, dur, play, lab, lab])
    return rows


def make_mini_data(root) -> Path:
    """Create root/{video_features_basic,log_standard_*}.csv. Returns root."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    with open(root / "video_features_basic_pure.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["video_id", "author_id"])
        for v in range(N_VIDEOS):
            w.writerow([v, v % N_AUTHORS])

    # file 1 -> train
    with open(root / "log_standard_4_08_to_4_21_pure.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(_LOG_HEADER)
        w.writerows(_rows_for(_TRAIN_DATE, videos_per_user=6, offset=0))

    # file 2 -> valid + test (both date ranges live here)
    with open(root / "log_standard_4_22_to_5_08_pure.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(_LOG_HEADER)
        rows = _rows_for(_VALID_DATE, videos_per_user=5, offset=1)
        rows += _rows_for(_TEST_DATE, videos_per_user=5, offset=2)
        w.writerows(rows)

    return root
