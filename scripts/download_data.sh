#!/usr/bin/env bash
# Download KuaiRand-Pure into ./KuaiRand-Pure/. Run from anywhere.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ -f "KuaiRand-Pure/data/log_standard_4_08_to_4_21_pure.csv" ]; then
  echo "Data already present at $(pwd)/KuaiRand-Pure/data — nothing to do."
  exit 0
fi

URL="https://zenodo.org/records/10439422/files/KuaiRand-Pure.tar.gz"
if [ ! -f "KuaiRand-Pure.tar.gz" ]; then
  echo "Downloading (~46 MB)..."
  curl -L -o KuaiRand-Pure.tar.gz "$URL"
fi
tar xzf KuaiRand-Pure.tar.gz   # -> ./KuaiRand-Pure/data/
echo "Done. Data in $(pwd)/KuaiRand-Pure/data"
