"""Bridge to the frozen starter kit at the repo root.

``evaluate.py`` is the frozen scoring spec and is never edited. This module
puts the repo root on ``sys.path`` once and re-exports the pieces our code is
allowed to call, so the rest of ``tippytop`` never imports the kit by fragile
relative paths.

There is exactly ONE copy of the kit, shared with ``autoresearch_lg`` (which
seeds each experiment folder from these same files). A second vendored copy
would mean two ``evaluate.py`` files, and ``evaluate.py`` *is* the task spec.

Nothing here should ever modify kit behaviour — only expose it.
"""
from pathlib import Path
import sys

# The kit lives at the repo root — ONE copy, shared with autoresearch_lg,
# which seeds every experiment folder from these same files. Keeping a second
# vendored copy would mean two evaluate.py files, and evaluate.py *is* the
# task spec. See ARCHITECTURE.md.
REPO_ROOT = Path(__file__).resolve().parents[2]
KIT_DIR = REPO_ROOT
DEFAULT_DATA_DIR = REPO_ROOT / "KuaiRand-Pure" / "data"

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
