"""The contract every generated solution.py must satisfy, plus the iter-0 seed."""
from __future__ import annotations

# Shown to the LLM verbatim in prompts.
SOLUTION_CONTRACT = """\
Your output is a single, self-contained Python script `solution.py`, run as:

    python solution.py --data_dir <path> --split <valid|test> --out <scores.csv>

Requirements:
1. Train ONLY on the 'train' split. You may early-stop on 'valid'. You must NEVER
   read, score, or otherwise touch the 'test' split — doing so is disqualifying.
2. Produce exactly one score per row of the requested --split, in the row order of
   the loaded split, and write a submission CSV with the helper:

       import argparse
       from tippytop.data.dataset import load_dataset
       from tippytop.submission import write_submission
       a = argparse.ArgumentParser(); a.add_argument('--data_dir'); \
           a.add_argument('--split', default='valid'); a.add_argument('--out')
       args = a.parse_args()
       data = load_dataset(args.data_dir)
       rows = data.splits[args.split]           # raw rows for that split
       scores = ...                              # numpy array, len == len(rows)
       write_submission(args.out, rows, scores)

   (If you prefer to be fully self-contained, write the CSV yourself with header
    `row_id,user_id,video_id,score`, one line per row: row_id is the 0-based index,
    user_id = row[1], video_id = row[2], score any finite float.)
3. Use numpy only (no torch / sklearn / pandas). Keep it runnable on CPU in a few
   minutes. Be deterministic (fix seeds). Exit non-zero on error.

Row tuple layout: (date, user_id, video_id, author_id, tab, duration_ms, long_view).
The label is long_view (0/1). Ranking is WITHIN each user; only relative order of
scores per user matters. Metric = mean(GAUC, nDCG@5) on valid.
"""


def seed_solution_source() -> str:
    """Iteration 0: a known-good FM baseline script (valid primary ~0.60)."""
    return '''\
"""Seed solution: the FM baseline (reproduces the official baseline)."""
import argparse
from tippytop.data.dataset import load_dataset
from tippytop.training.runner import train_model
from tippytop.submission import write_submission


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--split", default="valid")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    data = load_dataset(args.data_dir)
    model = train_model("fm", data, seed=42)
    scores = model.predict(data, args.split)
    write_submission(args.out, data.splits[args.split], scores)


if __name__ == "__main__":
    main()
'''
