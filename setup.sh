#!/usr/bin/env bash
# One-shot environment setup for autoresearch_lg (the LangGraph harness).
# Creates .venv, installs the package + langgraph-cli, seeds .env, and
# checks for the KuaiRand-Pure data — same steps README.md documents by
# hand, just scripted. Safe to re-run (every step is idempotent).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

DATA_DIR="./KuaiRand-Pure/data"
DATA_FILES=(
    video_features_basic_pure.csv
    video_features_statistic_pure.csv
    user_features_pure.csv
    log_standard_4_08_to_4_21_pure.csv
    log_standard_4_22_to_5_08_pure.csv
    log_random_4_22_to_5_08_pure.csv
)

echo "== autoresearch_lg setup =="

# ---- 1. Python ------------------------------------------------------------
PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    PYTHON_BIN=python
fi
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "ERROR: no python3/python on PATH. Install Python 3.11+ first." >&2
    exit 1
fi
echo "-- using $($PYTHON_BIN --version)"

# ---- 2. Virtual environment ------------------------------------------------
if [ ! -d ".venv" ]; then
    echo "-- creating .venv"
    "$PYTHON_BIN" -m venv .venv
else
    echo "-- .venv already exists, reusing it"
fi

# venv layout differs: Scripts/ on native Windows Python, bin/ everywhere
# else (including a Linux-built venv run under WSL/git-bash).
if [ -f ".venv/Scripts/python.exe" ]; then
    VENV_PY=".venv/Scripts/python.exe"
elif [ -f ".venv/bin/python" ]; then
    VENV_PY=".venv/bin/python"
else
    echo "ERROR: .venv exists but no python found inside it — delete .venv and re-run." >&2
    exit 1
fi

# ---- 3. Install -------------------------------------------------------------
echo "-- installing autoresearch_lg + langgraph-cli (this can take a minute)"
"$VENV_PY" -m pip install -q --upgrade pip
"$VENV_PY" -m pip install -q -e . "langgraph-cli[inmem]"
echo "-- installed"

# ---- 4. .env ----------------------------------------------------------------
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "-- created .env from .env.example — add your ANTHROPIC_API_KEY and/or OPENAI_API_KEY before running anything"
else
    echo "-- .env already exists, leaving it alone"
fi

# ---- 5. KuaiRand-Pure data ---------------------------------------------------
missing=0
for f in "${DATA_FILES[@]}"; do
    [ -f "$DATA_DIR/$f" ] || missing=1
done
if [ "$missing" = "1" ]; then
    echo ""
    echo "-- KuaiRand-Pure data not found under $DATA_DIR. Download it:"
    echo "     wget https://zenodo.org/records/10439422/files/KuaiRand-Pure.tar.gz"
    echo "     tar xzf KuaiRand-Pure.tar.gz"
    echo "   (run those from the repo root — they produce ./KuaiRand-Pure/)"
else
    echo "-- KuaiRand-Pure data found"
fi

echo ""
echo "== done =="
echo "Activate the venv, then:"
if [ "$VENV_PY" = ".venv/Scripts/python.exe" ]; then
    echo "  source .venv/Scripts/activate"
else
    echo "  source .venv/bin/activate"
fi
echo "  python -m autoresearch_lg.cli setup --tag \$(date +%b%d | tr A-Z a-z)"
echo "  python -m autoresearch_lg.cli run   --tag <same tag>"
