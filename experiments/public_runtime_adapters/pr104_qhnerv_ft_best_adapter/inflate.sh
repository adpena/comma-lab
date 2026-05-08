#!/usr/bin/env bash
# Exact-replay adapter for public PR #104 (patattzel / qhnerv_ft_best).
#
# Public claimed score material (from checked-in README/report):
#   archive.zip:        178,637 bytes
#   SegNet distortion:  0.00070710
#   PoseNet distortion: 0.00016895
#   compression rate:   0.00475788
#   score:              0.23 rounded
#
# This adapter preserves the public PR104 decoder and archive bytes. It only
# adapts the canonical contest harness member-lookup contract so
# archive.zip -> inflate.sh -> upstream/evaluate.py can run through
# experiments/contest_auth_eval.py without changing scored payload bytes.
# PACT_RUNTIME_DEPENDENCY_ROOT=experiments/results/public_pr_intake_full/public_pr104_intake_20260505_auto/source/submissions/qhnerv_ft_best
set -euo pipefail

if [ "$#" -ne 3 ]; then
  echo "FATAL: expected DATA_DIR OUTPUT_DIR FILE_LIST arguments" >&2
  exit 2
fi

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../../.." && pwd)"
SOURCE_ROOT="$REPO_ROOT/experiments/results/public_pr_intake_full/public_pr104_intake_20260505_auto/source"
RUNTIME_ROOT="$SOURCE_ROOT/submissions/qhnerv_ft_best"
INFLATE_PY="$RUNTIME_ROOT/inflate.py"

DATA_DIR="$1"
OUTPUT_DIR="$2"
FILE_LIST="$3"

if [ ! -f "$INFLATE_PY" ]; then
  echo "FATAL: PR104 public inflate.py missing: $INFLATE_PY" >&2
  exit 3
fi

PYBIN="${PACT_PYTHON:-${PYTHON:-$REPO_ROOT/.venv/bin/python}}"
if [ ! -x "$PYBIN" ]; then
  echo "FATAL: managed Python is not executable: $PYBIN" >&2
  exit 4
fi

"$PYBIN" - <<'PY'
import importlib.util
import sys

for module in ("brotli", "numpy", "torch"):
    if importlib.util.find_spec(module) is None:
        raise SystemExit(f"FATAL: required PR104 runtime dependency missing: {module}")

import brotli
import torch

print(
    "[pr104-adapter] "
    f"torch={getattr(torch, '__version__', 'unknown')} "
    f"cuda_available={torch.cuda.is_available()} "
    f"brotli={getattr(brotli, '__version__', 'unknown')}"
)
PY

mkdir -p "$OUTPUT_DIR"

while IFS= read -r line; do
  [ -z "$line" ] && continue
  BASE="${line%.*}"
  BASE_BIN="${DATA_DIR}/${BASE}.bin"
  X_MEMBER="${DATA_DIR}/x"
  if [ -f "$BASE_BIN" ] && [ -f "$X_MEMBER" ]; then
    echo "FATAL: ambiguous PR104 payload members; both ${BASE_BIN} and ${X_MEMBER} exist" >&2
    exit 5
  fi
  if [ -f "$BASE_BIN" ]; then
    SRC="$BASE_BIN"
  elif [ -f "$X_MEMBER" ]; then
    SRC="$X_MEMBER"
  else
    echo "FATAL: neither ${BASE_BIN} nor ${X_MEMBER} exists" >&2
    exit 6
  fi
  DST="${OUTPUT_DIR}/${BASE}.raw"
  echo "[pr104-adapter] inflating ${SRC} -> ${DST}"
  PYTHONPATH="$RUNTIME_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
    "$PYBIN" "$INFLATE_PY" "$SRC" "$DST"
done < "$FILE_LIST"
