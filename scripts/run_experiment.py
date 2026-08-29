"""Thin wrapper: `python scripts/run_experiment.py --model fm [args]`.

Equivalent to `python -m tippytop run --model fm`. Exists so you can run without
installing the package (it puts ./src on sys.path first).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tippytop.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main(["run"] + sys.argv[1:]))
