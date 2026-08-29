#!/usr/bin/env bash
# Download KuaiRand-Pure into the vendored kit dir. Run from repo root.
set -euo pipefail
cd "$(dirname "$0")/../kuairand-starter-kit/kuairand-starter-kit"
url="https://zenodo.org/records/10439422/files/KuaiRand-Pure.tar.gz"
[ -f KuaiRand-Pure.tar.gz ] || wget "$url"
tar xzf KuaiRand-Pure.tar.gz   # -> ./KuaiRand-Pure/data/
echo "Done. Data in $(pwd)/KuaiRand-Pure/data"
