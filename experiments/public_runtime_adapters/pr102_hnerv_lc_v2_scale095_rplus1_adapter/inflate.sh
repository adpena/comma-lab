#!/usr/bin/env bash
# Exact-replay adapter for public PR #102 (EthanYangTW / hnerv_lc_v2_scale095_rplus1).
#
# Public claimed scores (from PR body):
#   archive.zip:        178,981 bytes
#   PoseNet distortion: 0.00003460 (CPU report) / 0.000033274 (CUDA scorer)
#   SegNet distortion:  0.00057602 (CPU report) / 0.000575697 (CUDA scorer)
#   Compression rate:   0.00476704
#   Final score:        0.1953791765 (CPU rounded) / 0.194986956 (CUDA, bronze medal)
#
# Architecture: built on @BradyMeighan's hnerv_lc_v2 (PR #100, itself built on
# PR #98 + PR #95). Archive payload UNCHANGED from PR #100 — only
# inference-time code constants changed:
#   - retuned latent correction scale 0.0100 → 0.0095
#   - zero-byte decode-side nudge: frame 0 red channel +1
#
# This adapter wraps the public PR's inflate.py through repo-managed
# .venv/bin/python — preserving archive bytes, swapping only the python
# interpreter and adding strict member-lookup contract handling.
# PACT_RUNTIME_DEPENDENCY_ROOT=experiments/results/public_pr_intake_full/public_pr102_intake_20260505_auto/source/submissions/hnerv_lc_v2_scale095_rplus1
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../../.." && pwd)"
SOURCE_ROOT="$REPO_ROOT/experiments/results/public_pr_intake_full/public_pr102_intake_20260505_auto/source"
INFLATE_PY="$SOURCE_ROOT/submissions/hnerv_lc_v2_scale095_rplus1/inflate.py"

DATA_DIR="${1:?data dir required}"
OUTPUT_DIR="${2:?output dir required}"
FILE_LIST="${3:?file list required}"

if [ ! -f "$INFLATE_PY" ]; then
  echo "FATAL: PR102 public inflate.py missing: $INFLATE_PY" >&2
  exit 2
fi

PYBIN="${PYTHON:-$REPO_ROOT/.venv/bin/python}"
if [ ! -x "$PYBIN" ]; then
  echo "FATAL: managed Python is not executable: $PYBIN" >&2
  exit 4
fi

"$PYBIN" - <<'PY'
import brotli
print(f"[pr102-adapter] brotli={getattr(brotli, '__version__', 'unknown')}")
PY

mkdir -p "$OUTPUT_DIR"

while IFS= read -r line; do
  [ -z "$line" ] && continue
  BASE="${line%.*}"
  BASE_BIN="${DATA_DIR}/${BASE}.bin"
  X_MEMBER="${DATA_DIR}/x"
  if [ -f "$BASE_BIN" ] && [ -f "$X_MEMBER" ]; then
    echo "FATAL: ambiguous PR102 payload members; both ${BASE_BIN} and ${X_MEMBER} exist" >&2
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
  echo "[pr102-adapter] inflating ${SRC} -> ${DST}"
  PYTHONPATH="$SOURCE_ROOT/submissions/hnerv_lc_v2_scale095_rplus1/src${PYTHONPATH:+:$PYTHONPATH}" \
    "$PYBIN" "$INFLATE_PY" "$SRC" "$DST"
done < "$FILE_LIST"
