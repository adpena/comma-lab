#!/usr/bin/env bash
# Exact-replay adapter for public PR #103 (rem2 / hnerv_lc_ac).
#
# Public claimed scores (from PR body):
#   archive.zip:        178,223 bytes
#   PoseNet distortion: 0.00003443
#   SegNet distortion:  0.00057638
#   Compression rate:   0.00474686
#   Final score:        0.19487 (silver medal)
#
# Architecture: lossless byte-level repack of @BradyMeighan's hnerv_lc_v2 (PR #100,
# itself built on PR #98 + PR #95). The substantive change vs PR #100 is
# arithmetic coding (constriction range coder) on the 8 largest weight tensors
# and the latent-hi byte stream, replacing pure brotli on those payloads. Other
# bytes saved via hardcoded section lengths inside inflate.py, adaptive lgwin in
# brotli, single-byte filename in zip, and merging 9 AC streams into one
# RangeEncoder.
#
# This adapter wraps the public PR's inflate.py through repo-managed
# .venv/bin/python — preserving archive bytes, swapping only the python
# interpreter and adding strict member-lookup contract handling.
# PACT_RUNTIME_DEPENDENCY_ROOT=experiments/results/public_pr_intake_full/public_pr103_intake_20260505_auto/source/submissions/hnerv_lc_ac
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../../.." && pwd)"
SOURCE_ROOT="$REPO_ROOT/experiments/results/public_pr_intake_full/public_pr103_intake_20260505_auto/source"
INFLATE_PY="$SOURCE_ROOT/submissions/hnerv_lc_ac/inflate.py"

DATA_DIR="${1:?data dir required}"
OUTPUT_DIR="${2:?output dir required}"
FILE_LIST="${3:?file list required}"

if [ ! -f "$INFLATE_PY" ]; then
  echo "FATAL: PR103 public inflate.py missing: $INFLATE_PY" >&2
  exit 2
fi

PYBIN="${PYTHON:-$REPO_ROOT/.venv/bin/python}"
if [ ! -x "$PYBIN" ]; then
  echo "FATAL: managed Python is not executable: $PYBIN" >&2
  exit 4
fi

"$PYBIN" - <<'PY'
import brotli
print(f"[pr103-adapter] brotli={getattr(brotli, '__version__', 'unknown')}")
try:
    import constriction
    print(f"[pr103-adapter] constriction={getattr(constriction, '__version__', 'unknown')}")
except ImportError as e:
    print(f"[pr103-adapter] WARNING constriction not available ({e}); arithmetic-coded streams will fail")
PY

mkdir -p "$OUTPUT_DIR"

while IFS= read -r line; do
  [ -z "$line" ] && continue
  BASE="${line%.*}"
  BASE_BIN="${DATA_DIR}/${BASE}.bin"
  X_MEMBER="${DATA_DIR}/x"
  if [ -f "$BASE_BIN" ] && [ -f "$X_MEMBER" ]; then
    echo "FATAL: ambiguous PR103 payload members; both ${BASE_BIN} and ${X_MEMBER} exist" >&2
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
  echo "[pr103-adapter] inflating ${SRC} -> ${DST}"
  PYTHONPATH="$SOURCE_ROOT/submissions/hnerv_lc_ac/src${PYTHONPATH:+:$PYTHONPATH}" \
    "$PYBIN" "$INFLATE_PY" "$SRC" "$DST"
done < "$FILE_LIST"
