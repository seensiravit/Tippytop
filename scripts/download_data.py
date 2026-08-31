"""Download KuaiRand-Pure into ./KuaiRand-Pure/. Works on every platform.

    python scripts/download_data.py

Why this exists alongside the .sh and .ps1 versions: a stock Windows box has no
`bash`, and PowerShell refuses to run an unsigned `.ps1` under the default
execution policy ("...is not digitally signed"). Both of those are environment
friction rather than real problems, and both disappear if the downloader is just
Python — which the project already requires.

Standard library only, so it runs before any dependency is installed.
"""
from __future__ import annotations

import argparse
import sys
import tarfile
import urllib.request
from pathlib import Path

URL = "https://zenodo.org/records/10439422/files/KuaiRand-Pure.tar.gz"
ARCHIVE = "KuaiRand-Pure.tar.gz"
SENTINEL = Path("KuaiRand-Pure") / "data" / "log_standard_4_08_to_4_21_pure.csv"


def _progress(done: int, block: int, total: int) -> None:
    if total <= 0:
        return
    pct = min(100.0, 100.0 * done * block / total)
    mb = total / 1e6
    print(f"\r  {pct:5.1f}%  of {mb:.0f} MB", end="", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=None,
                    help="where to unpack (default: the repository root)")
    ap.add_argument("--force", action="store_true",
                    help="re-download even if the data is already present")
    a = ap.parse_args()

    root = Path(a.root) if a.root else Path(__file__).resolve().parents[1]
    root.mkdir(parents=True, exist_ok=True)

    target = root / SENTINEL
    if target.exists() and not a.force:
        print(f"Data already present at {root / 'KuaiRand-Pure' / 'data'} — nothing to do.")
        return 0

    archive = root / ARCHIVE
    if not archive.exists() or a.force:
        print(f"Downloading KuaiRand-Pure (~46 MB) from Zenodo...")
        try:
            urllib.request.urlretrieve(URL, archive, reporthook=_progress)
            print()
        except Exception as e:                       # noqa: BLE001
            print(f"\nDownload failed: {type(e).__name__}: {e}", file=sys.stderr)
            print(f"You can fetch it manually from:\n  {URL}\n"
                  f"and place it at {archive}, then re-run this script.",
                  file=sys.stderr)
            return 1
    else:
        print(f"Using existing archive at {archive}")

    print("Extracting...")
    with tarfile.open(archive, "r:gz") as tf:
        # filter='data' blocks absolute paths and traversal outside the target.
        # Added in 3.12 and the default from 3.14; passed explicitly so the
        # behaviour is the same on every supported version.
        try:
            tf.extractall(root, filter="data")
        except TypeError:                            # Python < 3.12
            tf.extractall(root)

    if not target.exists():
        print(f"Extraction finished but {target} is missing — the archive layout "
              "may have changed.", file=sys.stderr)
        return 1

    print(f"Done. Data in {root / 'KuaiRand-Pure' / 'data'}")
    print("\nNext: python -m tippytop run --model random --no-log")
    print("Expect test primary ~ 0.4753 (+-0.001). If not, the harness is broken "
          "and nothing else is trustworthy yet.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
