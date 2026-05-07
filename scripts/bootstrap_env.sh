#!/usr/bin/env bash
# One-shot Python venv + dependencies + tsfile clone (README §3.1).
# Run from anywhere:  bash scripts/bootstrap_env.sh
# Or:                 ./scripts/bootstrap_env.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "Missing command: $1" >&2
    exit 1
  }
}

need_cmd python3
need_cmd git

python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)' || {
  echo "Need Python 3.9 or newer (found: $(python3 -V 2>&1))" >&2
  exit 1
}

if [[ ! -d .venv ]]; then
  python3 -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install -U pip
pip install -r requirements.txt

if [[ ! -d tsfile ]]; then
  git clone -b research/encoding-pack-size --single-branch https://github.com/apache/tsfile.git
else
  echo "Directory tsfile/ already exists; skip clone."
fi

echo ""
echo "Python venv ready: source .venv/bin/activate"
echo "TsFile tree:       $ROOT/tsfile"
echo ""
echo "For Java tests (README §3.2), install JDK 17 (or 11+) and Maven 3.6+."
if command -v java >/dev/null 2>&1; then
  java -version 2>&1 | head -n 1 || true
else
  echo "  java:  not found"
fi
if command -v mvn >/dev/null 2>&1; then
  mvn -version | head -n 1 || true
else
  echo "  mvn:   not found"
fi
