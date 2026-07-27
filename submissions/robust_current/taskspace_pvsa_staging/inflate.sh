#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/../../.." && pwd)"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
exec "${PYTHON:-python3}" "${HERE}/inflate.py" "$1" "$2" "$3"
