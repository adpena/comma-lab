#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel >/dev/null
pip install -e . >/dev/null
comma-lab bootstrap-upstream
comma-lab install-submission exact_current
comma-lab install-submission robust_current
printf '\nBootstrap complete. Next, run: bash start.sh\n'
