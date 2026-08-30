"""Thin wrapper: `python scripts/make_submission.py --model fm --split test --out ...`.

Equivalent to `python -m tippytop submit ...`. Puts ./src on sys.path so it runs
without installing the package.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tippytop.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main(["submit"] + sys.argv[1:]))
