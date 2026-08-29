"""Bridge to the vendored, frozen starter kit.

The starter kit under ``kuairand-starter-kit/kuairand-starter-kit/`` is kept
UNTOUCHED (``evaluate.py`` is the frozen scoring spec). This module puts it on
``sys.path`` once and re-exports the pieces our code is allowed to call, so the
rest of ``tippytop`` never imports the kit by fragile relative paths.

Nothing here should ever modify kit behaviour — only expose it.
"""
from pathlib import Path
import sys

# Tippytop/ (repo root) -> kuairand-starter-kit/kuairand-starter-kit/
REPO_ROOT = Path(__file__).resolve().parents[2]
KIT_DIR = REPO_ROOT / "kuairand-starter-kit" / "kuairand-starter-kit"
DEFAULT_DATA_DIR = KIT_DIR / "KuaiRand-Pure" / "data"

if not KIT_DIR.exists():  # pragma: no cover - defensive
    raise RuntimeError(f"Vendored starter kit not found at {KIT_DIR}")

if str(KIT_DIR) not in sys.path:
    sys.path.insert(0, str(KIT_DIR))

# Re-export the frozen spec + data layer. Import lazily-safe at module load.
from evaluate import evaluate            # noqa: E402  (frozen scoring — never wrap/alter)
from data import load, encode, FIELDS, SPLITS, LABEL  # noqa: E402

__all__ = [
    "evaluate", "load", "encode", "FIELDS", "SPLITS", "LABEL",
    "REPO_ROOT", "KIT_DIR", "DEFAULT_DATA_DIR",
]
