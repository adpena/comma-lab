#!/usr/bin/env bash
# Exact-replay adapter for public PR #101 (SajayR / hnerv_ft_microcodec).
#
# Public claimed score (from submission README):
#   archive.zip:        178,258 bytes
#   SegNet distortion:  0.00056018
#   PoseNet distortion: 0.00003286
#   compression rate:   0.00474779
#   score:              0.19284
#
# Built on top of PR #95 + PR #98. Adds a self-contained entropy repack of the
# decoder, temporal latents, correction sidecar, and related payload
# optimizations.
#
# This adapter wraps the public PR's inflate.py through repo-managed
# .venv/bin/python — preserving archive bytes, swapping only the python
# interpreter and adding strict member-lookup contract handling.
# PACT_RUNTIME_DEPENDENCY_ROOT=experiments/results/public_pr_intake_full/public_pr101_intake_20260505_auto/source/submissions/hnerv_ft_microcodec
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../../.." && pwd)"
SOURCE_ROOT="$REPO_ROOT/experiments/results/public_pr_intake_full/public_pr101_intake_20260505_auto/source"
INFLATE_PY="$SOURCE_ROOT/submissions/hnerv_ft_microcodec/inflate.py"

DATA_DIR="${1:?data dir required}"
OUTPUT_DIR="${2:?output dir required}"
FILE_LIST="${3:?file list required}"

if [ ! -f "$INFLATE_PY" ]; then
  echo "FATAL: PR101 public inflate.py missing: $INFLATE_PY" >&2
  exit 2
fi

PYBIN="${PYTHON:-$REPO_ROOT/.venv/bin/python}"
if [ ! -x "$PYBIN" ]; then
  echo "FATAL: managed Python is not executable: $PYBIN" >&2
  exit 4
fi

"$PYBIN" - <<'PY'
import brotli
print(f"[pr101-adapter] brotli={getattr(brotli, '__version__', 'unknown')}")
PY

mkdir -p "$OUTPUT_DIR"

while IFS= read -r line; do
  [ -z "$line" ] && continue
  BASE="${line%.*}"
  BASE_BIN="${DATA_DIR}/${BASE}.bin"
  X_MEMBER="${DATA_DIR}/x"
  if [ -f "$BASE_BIN" ] && [ -f "$X_MEMBER" ]; then
    echo "FATAL: ambiguous PR101 payload members; both ${BASE_BIN} and ${X_MEMBER} exist" >&2
    exit 5
  fi
  if [ -f "$BASE_BIN" ]; then
    SRC="$BASE_BIN"
  elif [ -f "$X_MEMBER" ]; then
    SRC="$X_MEMBER"
  else
    echo "FATAL: neither ${BASE_BIN} nor ${X_MEMBER} exists" >&2
    exit 3
  fi
  DST="${OUTPUT_DIR}/${BASE}.raw"
  echo "[pr101-adapter] inflating ${SRC} -> ${DST}"
  PYTHONPATH="$SOURCE_ROOT/submissions/hnerv_ft_microcodec/src${PYTHONPATH:+:$PYTHONPATH}" \
    "$PYBIN" "$INFLATE_PY" "$SRC" "$DST"
done < "$FILE_LIST"
