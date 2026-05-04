#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${PACT_REPO_ROOT:-$(pwd)}"
SUBMISSION_DIR="${PACT_PUBLIC_SOURCE_SUBMISSION_DIR:-${REPO_ROOT}/experiments/results/public_pr106_belt_and_suspenders_intake_20260504_codex/source/submissions/belt_and_suspenders}"
PYTHON_BIN="${PYTHON:-${REPO_ROOT}/.venv/bin/python}"

if [ ! -x "$PYTHON_BIN" ]; then
  PYTHON_BIN="$(command -v python3 || command -v python)"
fi
if [ ! -f "${SUBMISSION_DIR}/inflate.py" ]; then
  echo "ERROR: submission inflate.py not found at ${SUBMISSION_DIR}" >&2
  exit 1
fi

DATA_DIR="$1"
OUTPUT_DIR="$2"
FILE_LIST="$3"
mkdir -p "$OUTPUT_DIR"

export PYTHONPATH="${SUBMISSION_DIR}/src:${SUBMISSION_DIR}:${PYTHONPATH:-}"
"$PYTHON_BIN" - <<'PY'
import brotli  # noqa: F401
PY

while IFS= read -r line; do
  [ -z "$line" ] && continue
  BASE="${line%.*}"
  SRC="${DATA_DIR}/${BASE}.bin"
  DST="${OUTPUT_DIR}/${BASE}.raw"
  if [ ! -f "$SRC" ] && [ -f "${DATA_DIR}/x" ]; then
    SRC="${DATA_DIR}/x"
  fi
  [ ! -f "$SRC" ] && echo "ERROR: ${SRC} not found" >&2 && exit 1
  "$PYTHON_BIN" "${SUBMISSION_DIR}/inflate.py" "$SRC" "$DST"
done < "$FILE_LIST"
